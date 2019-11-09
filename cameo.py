"""
@author:shuxitech
@file: cameo.py
@time: 2019/11/08
@describe: 
"""
import cv2
from managers import WindowManger, CaptureManger
import filters


class Cameo(object):

    def __init__(self):
        self._windowManger = WindowManger('Cameo', self.oneKeypress)
        self._captureManger = CaptureManger(cv2.VideoCapture(0), self._windowManger, True)
        self._filter = filters.SharpenFilter()  # 锐化

    def run(self):
        self._windowManger.createWindow()
        while self._windowManger.isWindowCreated:
            self._captureManger.enterFrame()
            frame = self._captureManger.frame
            filters.strokeEdges(frame, frame)
            self._filter.apply(frame, frame)

            self._captureManger.exitFrame()
            self._windowManger.processEvents()

    def oneKeypress(self, keycode):
        if keycode == 32:
            self._captureManger.writeImage('screenshot.png')
        elif keycode == 9:
            if not self._captureManger.isWritingVideo:
                self._captureManger.startWritingVideo('screencast.avi')
            else:
                self._captureManger.stopWritingVideo()
        elif keycode == 27:
            self._windowManger.destroyWindow()


if __name__ == '__main__':
    Cameo().run()
