import numpy as np
import cv2

if __name__ == '__main__':
    img = cv2.imread('img.png')
    cv2.imshow('window_name', img)
    cv2.waitKey(0)
    cv2.destroyWindow()

