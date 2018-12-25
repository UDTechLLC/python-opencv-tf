import cv2
import time
import imutils
import numpy as np


# TODO: camera parameters, like resolution, latency, fps (framerate, frame interval)
# TODO: best video processing
# TODO: image preprocessing, like decreasing/removing noise/shadows/blobs, enhancement, background substraction
class CaptureManager(object):
    def __init__(self, capture):
        self._capture = capture
        self._entered_frame = False

        self._original_frame = None
        self._processed_frame = None
        self._roi_frame = None

    @property
    def original_frame(self):
        if self._entered_frame and self._original_frame is None:
            _, self._original_frame = self._capture.retrieve()
            self._original_frame = imutils.rotate_bound(self._original_frame, -90)
        return self._original_frame

    @original_frame.setter
    def original_frame(self, value):
        self._original_frame = value

    @property
    def processed_frame(self):
        return self._processed_frame

    @processed_frame.setter
    def processed_frame(self, value):
        self._processed_frame = value

    @property
    def roi_frame(self):
        return self._roi_frame

    @roi_frame.setter
    def roi_frame(self, value):
        self._roi_frame = value

    def enter_frame(self):
        """Capture the next frame, if any."""
        # But first, check that any previous frame was exited.
        assert not self._entered_frame, \
            'previous enter_frame() had no matching exit_frame()'

        if self._capture is not None:
            self._entered_frame = self._capture.grab()

    def exit_frame(self):
        """Draw to the window. Write to files. Release the frame."""
        # Check whether any grabbed frame is retrievable.
        # The getter may retrieve and cache the frame.
        if self.original_frame is None:
            self._entered_frame = False
            return

        self._entered_frame = False

    def release_frame(self):
        # Release the frame.
        self._original_frame = None
        self._processed_frame = None
        # self._roi = None

    def read(self):
        success, image = self._capture.read()
        # height, width, channels = image.shape
        # print(0, height, width)

        image = imutils.rotate_bound(image, -90)
        x1, x2, y1, y_top, y_bottom, image_thresh = self.detect_rect_roi(image)

        return success, image_thresh

    @staticmethod
    def detect_rect_roi(image):
        height_new, width_new, channels_new = image.shape

        y_top = 330
        y_bottom = 1020
        thresh_level = 30
        # cv2.line(image, (0, y_top), (width_new, y_top), (0, 0, 255), 3)
        # cv2.line(image, (0, y_bottom), (width_new, y_bottom), (0, 0, 255), 3)

        image_crop = image[y_top:y_bottom, 0:width_new]
        image_blur = cv2.medianBlur(image_crop, 15)
        image_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
        image_thresh = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        height_thresh, width_thresh = image_thresh.shape

        cx = np.count_nonzero(image_thresh, axis=0)
        cxr = cx[::-1]
        cy = np.count_nonzero(image_thresh, axis=1)

        x1 = 0
        for x in np.ndenumerate(cx):
            if x[1] < height_thresh - thresh_level:
                # print(x[0][0], x[1])
                x1 = x[0][0]
                # cv2.line(image_thresh, (x[0][0], 0), (x[0][0], 1020 - 330), (0, 0, 0), 3)
                break

        x2 = 0
        for x in np.ndenumerate(cxr):
            if x[1] < height_thresh - thresh_level:
                # print(x[0][0], x[1])
                x2 = width_thresh - x[0][0]
                # cv2.line(image_thresh,
                #          (width_thresh - x[0][0], 0), (width_thresh - x[0][0], 1020 - 330), (0, 0, 0), 3)
                break

        y1 = 0
        for y in np.ndenumerate(cy):
            if y[1] < width_thresh - thresh_level/3:
                # print(y[0][0], y[1])
                y1 = y[0][0]
                # cv2.line(image_thresh, (0, y[0][0]), (width_thresh, y[0][0]), (0, 0, 0), 3)
                break

        # cv2.rectangle(image, (x1, y1 + y_top), (x2, y_bottom), (0, 255, 0), 5)
        cv2.rectangle(image_thresh, (x1, y1), (x2, height_thresh), (0, 0, 0), 5)

        return x1, x2, y1, y_top, y_bottom, image_thresh


class CaptureManagerWriter(CaptureManager):
    def __init__(self, capture):
        CaptureManager.__init__(self, capture)

        self._image_filename = None
        self._video_filename = None
        self._video_encoding = None
        self._video_writer = None

        self._start_time = None
        self._frames_elapsed = int(0)
        self._fps_estimate = None

    @property
    def is_writing_image(self):
        return self._image_filename is not None

    @property
    def is_writing_video(self):
        return self._video_filename is not None

    def exit_frame(self):
        CaptureManager.exit_frame(self)

        # Update the FPS estimate and related variables.
        if self._frames_elapsed == 0:
            self._start_time = time.time()
        else:
            time_elapsed = time.time() - self._start_time
            self._fps_estimate = self._frames_elapsed / time_elapsed
        self._frames_elapsed += 1

    def release_frame(self):
        # Write to the image file, if any.
        if self.is_writing_image:
            cv2.imwrite(self._image_filename, self._original_frame)
            self._image_filename = None

        # Write to the video file, if any.
        self._write_video_frame()

        CaptureManager.release_frame(self)

    def write_image(self, filename):
        """Write the next exited frame to an image file."""
        self._image_filename = filename

    def start_writing_video(
            self, filename,
            encoding=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')):
        """Start writing exited frames to a video file."""
        self._video_filename = filename
        self._video_encoding = encoding

    def stop_writing_video(self):
        """Stop writing exited frames to a video file."""
        self._video_filename = None
        self._video_encoding = None
        self._video_writer = None

    def _write_video_frame(self):
        if not self.is_writing_video:
            return

        if self._video_writer is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0.0:
                # The capture's FPS is unknown so use an estimate.
                if self._frames_elapsed < 20:
                    # Wait until more frames elapse so that the
                    # estimate is more stable.
                    return
                else:
                    fps = self._fps_estimate
            size = (int(self._capture.get(
                cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._capture.get(
                        cv2.CAP_PROP_FRAME_HEIGHT)))
            self._video_writer = cv2.VideoWriter(
                self._video_filename, self._video_encoding,
                fps, size)

        self._video_writer.write(self._original_frame)


def testCaptureManager():
    # DroidCam URL
    # url = 'http://192.168.55.129:4747/video'
    url = 0
    capture = cv2.VideoCapture(url)
    cm = CaptureManager(capture)

    cv2.namedWindow('CaptureManager Test', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('CaptureManager Test', 720, 500)

    while True:
        cm.enter_frame()
        frame = cm.original_frame

        cm.exit_frame()

        cv2.imshow('CaptureManager Test', frame)

        cm.release_frame()

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # when everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()


def testCaptureManagerWriter():
    # DroidCam URL
    # url = 'http://192.168.55.129:4747/video'
    url = 0
    capture = cv2.VideoCapture(url)
    cm = CaptureManagerWriter(capture)

    cv2.namedWindow('CaptureManager Test', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('CaptureManager Test', 720, 500)

    while True:
        cm.enter_frame()
        frame = cm.original_frame

        cm.exit_frame()

        cv2.imshow('CaptureManager Test', frame)

        cm.release_frame()

        keycode = cv2.waitKey(1)

        if keycode == 32:  # space
            cm.write_image('screenshot.png')
        elif keycode == 9:  # tab
            if not cm.is_writing_video:
                cm.start_writing_video('screencast.avi')
            else:
                cm.stop_writing_video()
        elif keycode == 27:  # escape
            break

    # when everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # testCaptureManager()
    testCaptureManagerWriter()
