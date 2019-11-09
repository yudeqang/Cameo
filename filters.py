"""
@author:shuxitech
@file: filters.py
@time: 2019/11/08
@describe: 
"""
import cv2
import numpy as np
import utils


def strokeEdges(src, dst, blurKsize=3, edgeKsize=5):
    if blurKsize >= 3:
        blurredSrc = cv2.medianBlur(src, blurKsize)  # 模糊
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # 转为灰度
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)
    normalizedInverseAlpha = (1.0 / 255) * (255-graySrc)  # 归一化
    channels = cv2.split(src)  # 通道拆分
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha  # 将边缘变黑
    cv2.merge(channels, dst)  # 通道合并


class VConvolution_Filter(object):
    """卷积滤波器"""
    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        cv2.filter2D(src, -1, self._kernel, dst)


class SharpenFilter(VConvolution_Filter):
    """锐化滤波器"""
    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        super().__init__(kernel)


class FindEdgesFilter(VConvolution_Filter):
    """边缘检测"""
    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]])
        super().__init__(kernel)


class BlurFilter(VConvolution_Filter):
    """模糊滤波器"""
    def __init__(self):
        kernel = np.array([[0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04]])
        super().__init__(kernel)


class EmbossFilter(VConvolution_Filter):
    """浮雕效果"""
    def __init__(self):
        kernel = np.array([[-2, -1, 0],
                           [-1,  1, 1],
                           [0, 1, 2]])
        super().__init__(kernel)


def canny_edge(src):
    dst = cv2.Canny(src, 200, 300)
    return dst


if __name__ == '__main__':
    src = cv2.imread(r'test1.jpg')
    cv2.imshow('input', src)
    strokeEdges(src, src)
    cv2.imshow('sE', src)

    FindEdgesFilter().apply(src, src)
    cv2.imshow('edgeKernel', src)

    cv2.imshow('canny', canny_edge(src))
    cv2.waitKey()
    cv2.destroyAllWindows()
