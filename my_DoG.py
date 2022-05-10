import cv2
import numpy as np

# library add
import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_filtering import my_filtering


def get_DoG_filter(fsize, sigma=1):
    ###################################################
    # TODO                                            #
    # DoG mask 완성                                    #
    ###################################################
    y, x = np.mgrid[-(fsize//2):(fsize//2) + 1, -(fsize//2):(fsize//2) + 1]

    DoG_x = (-x / (sigma ** 2)) * np.exp((-(x ** 2 + y ** 2) / 2 * (sigma ** 2)))
    DoG_y = (-y / (sigma ** 2)) * np.exp((-(x ** 2 + y ** 2) / 2 * (sigma ** 2)))

    return DoG_x, DoG_y

def my_padding(src, pad_shape, pad_type='zero'):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h+2*p_h, w+2*p_w))
    pad_img[p_h:p_h+h, p_w:p_w+w] = src

    if pad_type == 'repetition':
        print('repetition padding')

        #up
        pad_img[:p_h, p_w:p_w+w] = src[0, :]
        #down
        pad_img[p_h+h:, p_w:p_w+w] = src[h-1, :]

        #left
        pad_img[:,:p_w] = pad_img[:, p_w:p_w + 1]
        #right
        pad_img[:, p_w+w:] = pad_img[:, p_w+w-1:p_w+w]

    else:
        print('zero padding')

    return pad_img

def my_filtering(src, mask, pad_type='zero'):
    (h, w) = src.shape
    src_pad = my_padding(src, (mask.shape[0]//2, mask.shape[1]//2), pad_type)
    dst = np.zeros((h, w))

    #########################################################
    # TODO                                                  #
    # dst 완성                                              #
    # dst : filtering 결과 image                            #
    #########################################################

    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(cv2.multiply(src_pad[row:row + mask.shape[0], col:col + mask.shape[1]], mask))
            # if dst[row, col] > 255:     # value 가 255를 초과하는 경우 예외처리
            #     dst[row, col] = 255
            # elif dst[row, col] < 0:     # value 가 0 미만인 경우 예외처리
            #     dst[row, col] = 0

    # dst = (dst+0.5).astype(np.uint8)

    return dst

def main():
    src = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)
    DoG_x, DoG_y = get_DoG_filter(fsize=3, sigma=1)

    dst_x = my_filtering(src, DoG_x, 'zero')
    dst_y = my_filtering(src, DoG_y, 'zero')

    ###################################################
    # TODO                                            #
    # dst_x, dst_y 를 사용하여 magnitude 계산            #
    ###################################################
    # dst = np.abs(dst_x) + np.abs(dst_y)
    dst = np.sqrt((dst_x ** 2) + (dst_y ** 2))
    cv2.imshow('dst_x', dst_x / 255)
    cv2.imshow('dst_y', dst_y / 255)
    cv2.imshow('dst', dst / 255)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
