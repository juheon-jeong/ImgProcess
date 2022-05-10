import numpy as np
import cv2
import matplotlib.pyplot as plt

def my_calcHist(img):
    """
    :param img: 흑백 이미지
    :return: hist 1차원 배열
    """
    hist = np.zeros(256)

    for items in img:
        for item in items:
            hist[item] += 1
    return hist

def my_PDF2CDF(pdf):
    #  나눈 값을 누적시킨다
    """
    :param pdf: 1차원 배열
    :return: 1차원 배열
    """
    temp = pdf[0]
    for i in range(1, len(pdf)):
        pdf[i] = temp + pdf[i]
        temp = pdf[i]

    return pdf

def my_normalize_hist(hist, pixel_num):
    # histogram을 총 픽셀 수로 나눈다.
    """
    :param hist: 1차원 배열
    :param pixel_num: 총 픽셀 수
    :return: 1차원 배열
    """

    return hist/pixel_num

def my_denormallize(normalized, gray_level):
    #  누적시킨 값에 gray level을 곱한다.
    """
    :param normalized: 1차원 배열
    :param gray_level: max gray level
    :return: 1차원 배열
    """
    return normalized * gray_level

def my_calcHist_equalization(denormalized, hist):
    #  구해진 정수값을 사용하여 histogram equalization 결과를 return
    """
    :param denormalized: 1차원 배열
    :param hist: 1차원 배열
    :return: 1차원 배열
    """
    hist_equal = np.zeros(len(denormalized))

    for i in range(len(denormalized)):
        hist_equal[denormalized[i]] += hist[i]

    return hist_equal

def my_equal_img(src, gray_level):
    """
    :param src: 흑백 이미지
    :param gray_level: 1차원 배열
    :return: dst : 결과 이미지
    """
    (h, w) = src.shape
    dst = np.zeros((h,w)).astype(np.uint8)

    for i in range(h):
        for j in range(w):
            dst[i][j] = gray_level[src[i][j]]

    return dst

#input_image의  equalization된 histogram & image 를 return
def my_hist_equal(src):
    (h, w) = src.shape
    max_gray_level = 255
    histogram = my_calcHist(src)
    normalized_histogram = my_normalize_hist(histogram, h * w)
    normalized_output = my_PDF2CDF(normalized_histogram)
    denormalized_output = my_denormallize(normalized_output, max_gray_level)
    output_gray_level = denormalized_output.astype(int)
    hist_equal = my_calcHist_equalization(output_gray_level, histogram)

    ### dst : equalization 결과 image
    dst = my_equal_img(src, output_gray_level)
    plt.plot(output_gray_level, color = 'r')
    plt.title('mapping function')
    plt.xlabel('input')
    plt.ylabel('output')
    plt.show()
    return dst, hist_equal

if __name__ == '__main__':
    # Test on simple matrix

    test_img = np.array(
        [
            [0, 1, 1, 1, 2],
            [2, 3, 3, 3, 3],
            [3, 3, 3, 4, 4],
            [4, 4, 4, 4, 4],
            [4, 5, 5, 5, 7]
        ], dtype=np.uint8)
    hist = my_calcHist(test_img)
    dst, hist_equal = my_hist_equal(test_img)

    test_img_to_show = cv2.resize(test_img, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('equalizetion before image', test_img_to_show)
    test_dst_to_show = cv2.resize(dst, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('equalizetion after image', test_dst_to_show)

    plt.figure(1)
    plt.title('my histogram')
    plt.bar(np.arange(len(hist)), hist, width=0.5, color='g')

    plt.figure(2)
    plt.title('my histogram equalization')
    plt.bar(np.arange(len(hist_equal)), hist_equal, width=0.5, color='g')

    plt.show()
    cv2.waitKey()
    cv2.destroyAllWindows()

    # Test on real image
    test_img = cv2.imread('fruits.jpg', cv2.IMREAD_GRAYSCALE)
    hist = my_calcHist(test_img)
    dst, hist_equal = my_hist_equal(test_img)

    cv2.imshow('equalizetion before image', test_img)
    cv2.imshow('equalizetion after image', dst)

    plt.figure(1)
    plt.title('my histogram')
    plt.bar(np.arange(len(hist)), hist, width=0.5, color='g')

    plt.figure(2)
    plt.title('my histogram equalization')
    plt.bar(np.arange(len(hist_equal)), hist_equal, width=0.5, color='g')

    plt.show()
    cv2.waitKey()
    cv2.destroyAllWindows()
