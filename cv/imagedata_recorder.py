import cv2
import numpy as np
import time
import os
from cv.capture_manager import CaptureManager
from threading import Thread


class ImageDataRecorder(object):
    def __init__(self):
        self._record_face = 0
        self._record_count = 0
        self._stopped = True
        self._processed = False

        self._amount_frames = 0
        self._success_finding_contours = 0
        self._original_capture = None

        self._images = dict()

        self._width_zone = 0
        self._ratio_distance = 6.75
        self._margin = 50
        self._gap_bottom = 200

        # DroidCam URL
        # url = 'http://192.168.55.129:4747/video'
        url = 0
        capture = cv2.VideoCapture(url)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self._capture_manager = CaptureManager(capture)

    @property
    def original_capture(self):
        return self._original_capture

    @property
    def images(self):
        return self._images

    # from CaptureManager
    @property
    def processing_capture(self):
        return self._capture_manager.processed_frame

    @property
    def roi_capture(self):
        return self._capture_manager.roi_frame

    def release_frame(self):
        self._capture_manager.release_frame()

    # Thread
    @property
    def active(self):
        return not self._stopped

    @property
    def processed(self):
        return self._processed

    def set_width(self, value):
        self._width_zone = int(float(value) * self._ratio_distance)
        return self

    def start(self, face=0, count=100):
        print('Start recording')

        self._record_face = face
        self._record_count = count
        self._stopped = False

        # TODO: Thread Refactoring
        Thread(target=self.get, args=()).start()
        return self

    def processing(self, capture):
        # print('processing from Parent')
        self._original_capture = capture
        self._amount_frames += 1

        x1, x2, y1, y_top, y_bottom, image_thresh = self._capture_manager.detect_rect_roi(self._original_capture)

        self._capture_manager.roi_frame = self._original_capture.copy()
        self._capture_manager.roi_frame = self._capture_manager.roi_frame[y1 + y_top - 5:y_bottom, x1 - 5:x2 + 5]

        cv2.rectangle(self._original_capture, (x1 - 5, y1 + y_top - 5), (x2 + 5, y_bottom), (0, 0, 255), 5)
        self._images[self._record_face * 10 + self._success_finding_contours] = self._capture_manager.roi_frame

        self._success_finding_contours += 1

    def get(self):
        self._processed = True
        self._amount_frames = 0
        self._success_finding_contours = 0

        # time.sleep(0.1)

        while not self._stopped:
            # print('get: ' + str(self._success_finding_contours))
            self._capture_manager.enter_frame()

            original_capture = self._capture_manager.original_frame
            if original_capture is None:
                # self.stop()
                print('Original capture is None')
                continue

            self.processing(original_capture)

            self._capture_manager.exit_frame()
            self._capture_manager.release_frame()

            # time.sleep(0.1)

            if self._success_finding_contours >= self._record_count:
                print('Stop recording')
                self.stop()

        self._processed = False

    def get_video_frame(self):
        success, image = self._capture_manager.read()
        if image is None:
            return None

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def get_platform_frame(self):
        image = self._capture_manager.original_frame
        if image is None:
            return None

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def stop(self):
        self._record_face = 0
        self._record_count = 0
        self._stopped = True

    def write_dataset(self, dataset_name, class_name):
        image_folder = dataset_name + '/' + class_name + '/'
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        # write raw images
        for key in self._images.keys():
            image_filename = image_folder + 'image_' + str(key) + '.png'
            print(image_filename)
            cv2.imwrite(image_filename, self._images[key])

        return len(self._images)


class ImageDataRecorderProcessor(ImageDataRecorder):
    def __init__(self):
        ImageDataRecorder.__init__(self)

    def processing(self, capture):
        # print('processing from Child')

        self._original_capture = capture
        self._amount_frames += 1

        self._capture_manager.processed_frame = capture
        self._capture_manager.roi_frame = capture

        self._images[self._success_finding_contours * 10 + self._record_face] = self._capture_manager.roi_frame
        self._success_finding_contours += 1

        """
        if self._capture_manager.roi_frame is None:
            self._capture_manager.roi_frame = self._original_capture

        # print("Recording: " + str(self._success_finding_contours))
        # print("Count: " + str(self._record_count))

        self._capture_manager.processed_frame, roi_frame = \
            processing.process_and_detect(self._original_capture, self._window_manager)

        if roi_frame is not None:
            self._capture_manager.roi_frame = roi_frame
            self._data[self._success_finding_contours * 10 + self._record_face] = self._capture_manager.roi_frame

            self._success_finding_contours += 1

            color_yellow = (0, 255, 255)
            percent_success_finding = round((self._success_finding_contours / self._amount_frames) * 100, 2)
            cv2.putText(self._capture_manager.processed_frame, str(percent_success_finding) + "%",
                        (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color_yellow, 2)
            cv2.putText(self._capture_manager.processed_frame, str(self._success_finding_contours),
                        (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color_yellow, 2)
        """


def testImageDataRecorder():
    idr = ImageDataRecorder()
    idr.set_width(100)
    idr.start(face=0, count=20)

    while idr.processed:
        pass

    images_count = idr.write_dataset(dataset_name='../datasets/dataset_test', class_name='test_1')
    print('Were recorded ' + str(images_count) + ' images')


if __name__ == '__main__':
    testImageDataRecorder()
