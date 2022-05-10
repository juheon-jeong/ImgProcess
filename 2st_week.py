import cv2
import numpy as np

def my_bgr2gray(src):
    '''
    :param src:컬러 이미지
    :return dst1, dst2, dst3:흑백 이미지
    '''

    #cvtColor() 함수 이용
    dst1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    #########################
    # TODO                  #
    # dst2, dst3, dst4 채우기     #
    #########################
    #dst2는 B, G, R채널 각각 1/3씩 사용
    b = src[ : , : , 0]     # blue channel
    g = src[ : , : , 1]     # green channel
    r = src[ : , : , 2]     # red channel

    dst2 = b/3 + g/3 + r/3
    # #dst3은 B, G, R채널 중 하나의 채널만 사용(B,G,R중 원하는거 아무거나)
    dst3 = b            # Blue channel
    #
    # #dst2 반올림 np.round를 사용해도 무관
    dst2 = (dst2+0.5).astype(np.uint8)
    #
    # #dst4는 공식 사용
    # 소수점을 곱하므로 uint형으로 형변환을 해줘야함.
    dst4 = (0.2125*r + 0.7154*g + 0.0721*b).astype(np.uint8)

    return dst1, dst2, dst3, dst4


temp = ['fruits.jpg', 'Lena.png', 'baby.jpg']

for item in temp:
    src = cv2.imread(item)

    dst1, dst2, dst3, dst4 = my_bgr2gray(src)

    cv2.imshow('original', src)
    cv2.imshow('gray(cvtColor)', dst1)
    cv2.imshow('gray(1/3)', dst2)
    cv2.imshow('gray(one channel)', dst3)
    cv2.imshow('gray(phosphor)',dst4)

    cv2.waitKey()
    cv2.destroyAllWindows()