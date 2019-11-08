"""
@author:shuxitech
@file: managers.py
@time: 2019/11/07
@describe: 
"""
import cv2
import numpy as np
import time
from loguru import logger


class CaptureManger(object):

    def __init__(self, capture: cv2.VideoCapture, previewWindowManger=None, shouldMirrorPreview=False):
        self.previewWindowManger = previewWindowManger
        self.shodMirrorPreview = shouldMirrorPreview

        self._capture = capture
        self._channel = 0
        self._enteredFrame = False
        self._frame = None
        self._imageFilename = None
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

        self._startTime = None
        self._framesElapsed = 0
        self._fpsEstimate = None

    @property
    def channel(self):
        # 摄像头频道
        return self._channel

    @channel.setter
    def channel(self, value):
        # 设置摄像头频道
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve()
        return self._frame

    @property
    def isWritingImage(self) -> bool:
        # 是否写图片
        return self._imageFilename is not None

    @property
    def isWritingVideo(self) -> bool:
        # 是否写视频
        return self._videoFilename is not None

    def enterFrame(self):
        """获取下一帧"""
        assert not self._enteredFrame, '无匹配帧'
        if self._capture is not None:
            self._enteredFrame = self._capture.grab()

    def exitFrame(self):
        """在窗口显示、写入文件，释放帧"""
        if self.frame is None:
            self._enteredFrame = False
            return

        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time() - self._startTime
            self._fpsEstimate = self._framesElapsed / timeElapsed
        self._framesElapsed += 1

        if self.previewWindowManger is not None:
            if self.shodMirrorPreview:
                mirroredFrame = np.fliplr(self._frame).copy()
                self.previewWindowManger.show(mirroredFrame)
            else:
                self.previewWindowManger.show(self._frame)

        if self.isWritingImage:
            cv2.imwrite(self._imageFilename, self._frame)
            self._imageFilename = None

        self._writeVideoFrame()

        self._frame = None
        self._enteredFrame = False

    def writeImage(self, filename):
        """Write the next exited frame to an image file"""
        self._imageFilename = filename

    def startWritingVideo(self, filename, encoding=cv2.VideoWriter_fourcc('I', '4', '2', '0')):
        """Start writing exited frames to a video file"""
        self._videoFilename = filename
        self._videoEncoding = encoding

    def stopWritingVideo(self):
        """Stop writing exited frames to a video file"""
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

    def _writeVideoFrame(self):

        if not self.isWritingVideo:
            return

        if self._videoWriter is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps == 0.0:
                """The capture's FPS is unknown so use an estimate 不知道capture的FPS，所以使用一个预估值"""
                if self._framesElapsed < 20:
                    # Wait until more frames elapse so that the estimate is more stable
                    return
                else:
                    fps = self._fpsEstimate
                    size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

                    self._videoWriter = cv2.VideoWriter(
                        self._videoFilename, self._videoEncoding, fps, size
                    )  # 调用cv的videowrite方法写入视频


class WindowManger(object):

    def __init__(self, windowName, keypressCallback=None):
        self.keypressCallback = keypressCallback

        self._windowName = windowName
        self._isWindowCreated = False

    @property
    def isWindowCreated(self):
        return self._isWindowCreated

    def createWindow(self):
        """ """
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self, frame):
        """show window"""
        cv2.imshow(self._windowName, frame)

    def destroyWindow(self):
        logger.info("destroyWindow")
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvents(self):
        keycode = cv2.waitKey(-1)
        if self.keypressCallback is not None and keycode != -1:
            logger.info("capture key {}".format(keycode))
            # Discard any non-ASCII info encoded by GEK
            keycode &= 0xFF
            self.keypressCallback(keycode)
