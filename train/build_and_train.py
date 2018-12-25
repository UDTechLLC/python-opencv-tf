import os
import pickle
import configparser
import shutil
import numpy as np

import tensorflow as tf
from tensorflow.contrib import slim

import train.vgg as model_v

LR_START = 0.001
LR_END = LR_START / 1e4
MOMENTUM = 0.9
LOG_LOSS_EVERY = 10
CALC_ACC_EVERY = 500


class ModelWrapper:
    model = None
    pre_trained_restorer = None
    old_storage = None

    def __init__(self, network: str, pre_trained: str, storage: str, size: int, cls: int):
        self.network_name = network
        self.pre_trained = pre_trained
        if self.pre_trained == storage:
            os.mkdir("{}_{}".format(storage, str(cls)))
            self.storage = "{}_{}".format(storage, str(cls))
            self.old_storage = storage
        else:
            self.storage = storage
        self.cfg = configparser.ConfigParser()
        self.cfg.read('/home/apteka/code/apteka/prototype/package_entry/train/models_meta.ini')
        self.img_size = size
        self.tmp_nodes = {}

    def get_initial_meta(self):
        print(self.storage)
        if os.path.exists(self.storage):
            pass

    def set_model_meta(self, nodes_names: dict):
        self.cfg.set(self.network_name, 'input', nodes_names["input"])
        self.cfg.set(self.network_name, 'output', nodes_names["output"])
        self.cfg.set(self.network_name, 'image_size', str(self.img_size))
        self.cfg.set(self.network_name, 'last_checkpoints_dir', self.storage)
        with open('models_meta.ini', 'w') as configfile:
            self.cfg.write(configfile)

    def get_trainable_weights(self):
        vars_to_restore = tf.contrib.framework.get_variables_to_restore(
            exclude=['vgg_19/fc8'])
        print(vars_to_restore)
        self.pre_trained_restorer = tf.train.Saver(vars_to_restore)

    def build(self, num_classes: int, batch_size: int):
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            X = tf.placeholder(tf.float32, shape=(batch_size, self.img_size, self.img_size, 3), name="input")
            y_ = tf.placeholder(tf.float32, shape=(batch_size, num_classes), name="y")
            y_true_cls = tf.argmax(y_, dimension=1, name="y_true_cls")

            train_logits = model_v.vgg_19(X, num_classes, is_training=True)
            val_logits = model_v.vgg_19(X, num_classes, is_training=False)

            train_logits_s = tf.nn.softmax(train_logits)
            self.set_model_meta({"output": val_logits.name.split(":")[0], "input": X.name.split(":")[0]})

            self.tmp_nodes = {
                "X": X,
                "y_": y_,
                "y_true_cls": y_true_cls,
                "train_logits": train_logits,
                "val_logits": val_logits,
                "train_logits_s": train_logits_s
            }


class Trainer:
    def __init__(self):
        self.model = None
        self.data = None
        self.classes = None
        self.num_classes = 0
        self.epochs = 0
        self.batch_size = 0

        self.status = None
        self.step = 0
        self.amount = 0
        self.current_operation = ''
        self.count = 0

    def init(self, model: ModelWrapper, data, classes: list, epochs: int, batch_size: int):
        self.model = model
        self.data = data
        self.classes = classes
        self.num_classes = len(classes)
        self.epochs = epochs
        self.batch_size = batch_size

    def calc_accuracy(self, sess, val_logits, val_iters, y_true_cls, X, y_):
        acc_total = 0.0
        acc_denom = 0
        for i in range(val_iters):
            x_valid_batch, y_valid_batch, _, valid_cls_batch = self.data.valid.next_batch(self.batch_size)
            logits, y = sess.run([val_logits, y_true_cls], feed_dict={X: x_valid_batch, y_: y_valid_batch})
            y_pred = np.argmax(logits, axis=1)
            correct = np.count_nonzero(y == y_pred)
            acc_denom += y_pred.shape[0]
            acc_total += float(correct)
            print('Validating batch [{} / {}] correct = {}'.format(
                i, val_iters, correct))
        acc_total /= acc_denom
        return acc_total

    def accuracy_summary(self, sess, acc_value, iteration):
        acc_summary = tf.Summary()
        acc_summary.value.add(tag="accuracy", simple_value=acc_value)
        sess._hooks[1]._summary_writer.add_summary(acc_summary, iteration)

    def init_fn(self, scaffold, sess):
        ckpt = tf.train.get_checkpoint_state(self.model.pre_trained)
        if ckpt and ckpt.model_checkpoint_path:
            self.model.pre_trained_restorer.restore(sess, ckpt.model_checkpoint_path)
        else:
            self.model.pre_trained_restorer.restore(sess, self.model.pre_trained)

    def run_train(self):
        self.current_operation = 'training'
        self.step = 0
        self.status = 'Start'

        num_iteration = round(self.data.train.num_examples / self.batch_size) * self.epochs
        print('num_iteration', num_iteration)

        self.amount = num_iteration

        # iter_in_e = int(num_iteration / self.epochs)

        self.model.build(self.num_classes, batch_size=self.batch_size)
        self.model.get_trainable_weights()

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.model.tmp_nodes["train_logits_s"],
                                                       labels=self.model.tmp_nodes["y_"]))
        tf.summary.scalar('loss', loss)

        global_step = tf.train.get_or_create_global_step()
        lr = tf.train.polynomial_decay(LR_START, global_step, num_iteration, LR_END)
        tf.summary.scalar('learning_rate', lr)
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=MOMENTUM)
        training_op = slim.learning.create_train_op(
            loss, optimizer, global_step=global_step)
        scaffold = tf.train.Scaffold(init_fn=self.init_fn)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333, allow_growth=True)

        with tf.train.MonitoredTrainingSession(checkpoint_dir=self.model.storage,
                                               save_checkpoint_secs=600,
                                               save_summaries_steps=30,
                                               scaffold=scaffold) as sess:

            # todo UNPICKLE CORRECT ORDER OF CLASSES WHILE RECOGNIZE!
            """
            with open ('path/to/classes/file/classes.pckl', 'rb') as fp:
            classes = pickle.load(fp)
            """
            with open(os.path.join(self.model.storage, 'classes.pckl'), 'wb') as fp:
                pickle.dump(self.classes, fp)

            # remove old folder with data
            if self.model.old_storage:
                shutil.rmtree(self.model.old_storage)

            for iteration in range(num_iteration):

                x_batch, y_true_batch, _, cls_batch = self.data.train.next_batch(self.batch_size)

                # Gradient Descent
                _, loss_value = sess.run([training_op, loss], feed_dict={self.model.tmp_nodes["X"]: x_batch,
                                                                         self.model.tmp_nodes["y_"]: y_true_batch})

                # Loss logging
                if iteration % LOG_LOSS_EVERY == 0:
                    print('[{} / {}] Loss = {}'.format(
                        iteration, num_iteration, loss_value))

                # Accuracy logging
                if iteration % CALC_ACC_EVERY == 0:
                    # sess.run()
                    val_iters = round(self.data.valid.num_examples / self.batch_size)
                    acc_value = self.calc_accuracy(
                        sess,
                        self.model.tmp_nodes["val_logits"],
                        val_iters,
                        self.model.tmp_nodes["y_true_cls"],
                        self.model.tmp_nodes["X"],
                        self.model.tmp_nodes["y_"]
                    )
                    self.accuracy_summary(sess, acc_value, iteration)
                    print('[{} / {}] Validation accuracy = {}'.format(
                        iteration, num_iteration, acc_value))

                self.step += 1

        self.step = 0
        self.amount = 0
        self.status = 'Finish'
