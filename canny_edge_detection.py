import cv2
import numpy as np


def my_padding(src, pad_shape, pad_type='zero'):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w))
    pad_img[p_h:p_h + h, p_w:p_w + w] = src

    if pad_type == 'repetition':
        print('repetition padding')

        # up
        pad_img[:p_h, p_w:p_w + w] = src[0, :]
        # down
        pad_img[p_h + h:, p_w:p_w + w] = src[h - 1, :]
        # left
        pad_img[:, :p_w] = pad_img[:, p_w:p_w + 1]
        # right
        pad_img[:, p_w + w:] = pad_img[:, p_w + w - 1:p_w + w]

    else:
        print('zero padding')

    return pad_img


def my_filtering(src, mask, pad_type='zero'):
    (h, w) = src.shape
    src_pad = my_padding(src, (mask.shape[0] // 2, mask.shape[1] // 2), pad_type)
    dst = np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            val = np.sum(src_pad[row:row + mask.shape[0], col:col + mask.shape[1]] * mask)
            dst[row, col] = val

    return dst


def get_Gaussian_mask(fsize, sigma=1):
    y, x = np.mgrid[-(fsize // 2):(fsize // 2) + 1, -(fsize // 2):(fsize // 2) + 1]

    # 2차 gaussian mask 생성
    gaus2D = 1 / (2 * np.pi * sigma ** 2) * np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    # mask의 총 합 = 1
    gaus2D /= np.sum(gaus2D)

    return gaus2D


def get_sobel_mask():
    derivative = np.array([[-1, 0, 1]])
    blur = np.array([[1], [2], [1]])

    x = np.dot(blur, derivative)
    y = np.dot(derivative.T, blur.T)

    return x, y


def apply_gaussian_filter(src, fsize=3, sigma=1):
    #####################################################
    # TODO                                              #
    # src에 gaussian filter 적용                         #
    #####################################################
    Gaussian_mask = get_Gaussian_mask(fsize, sigma)
    dst = my_filtering(src, Gaussian_mask)
    return dst


def apply_sobel_filter(src):
    #####################################################
    # TODO                                              #
    # src에 sobel filter 적용                            #
    #####################################################
    Ix, Iy = get_sobel_mask()
    srcIx = my_filtering(src, Ix)
    srcIy = my_filtering(src, Iy)
    return srcIx, srcIy


def calc_magnitude(Ix, Iy):
    #####################################################
    # TODO                                              #
    # Ix, Iy 로부터 magnitude 계산                        #
    #####################################################
    magnitude = np.sqrt((Ix**2) + (Iy**2))

    return magnitude


def calc_angle(Ix, Iy, eps=1e-6):
    #####################################################
    # TODO                                              #
    # Ix, Iy 로부터 angle 계산                            #
    # numpy의 arctan 사용 O, arctan2 사용 X               #
    # 호도법이나 육십분법이나 상관 X                         #
    # eps     : Divide by zero 방지용                    #
    #####################################################
    angle = np.arctan(Iy / (Ix+eps))
    return np.rad2deg(angle)


def non_maximum_supression(magnitude, angle):
    ####################################################################################
    # TODO                                                                             #
    # Non-maximum-supression 수행                                                       #
    # 스켈레톤 코드는 angle이 육십분법으로 나타나져 있을 것으로 가정                             #
    ####################################################################################
    (h, w) = magnitude.shape
    # angle의 범위 : -90 ~ 90
    largest_magnitude = np.zeros((h, w))
    for row in range(1, h - 1):
        for col in range(1, w - 1):
            degree = angle[row, col]
            
            # 각도가 d일 때
            # d 각도의 픽셀과 동시에 180 + d 각도 방향의 픽셀과도 비교 해야함.
            # ex) 10도와 190도 -> 대략 우측과 좌측 픽셀
            # interpolation 방법은 linear로 구현
            if 0 <= degree and degree < 45:
                t = np.tan(np.deg2rad(degree))
                plus = magnitude[row, col + 1] * (1 - t) + magnitude[row + 1, col + 1] * t
                minus = magnitude[row, col - 1] * (1 - t) + magnitude[row - 1, col - 1] * t
            elif 45 <= degree and degree <= 90:
                t = 1 / np.tan(np.deg2rad(degree))
                plus = magnitude[row + 1, col] * (1 - t) + magnitude[row + 1, col + 1] * t
                minus = magnitude[row - 1, col - 1] * t + magnitude[row - 1, col] * (1 - t)
            elif -45 <= degree and degree < 0:
                t = np.tan(np.deg2rad(abs(degree)))
                plus = magnitude[row, col + 1] * (1 - t) + magnitude[row - 1, col + 1] * t
                minus = magnitude[row, col - 1] * (1 - t) + magnitude[row + 1, col - 1] * t
            elif -90 <= degree and degree < -45:
                t = 1 / np.tan(np.deg2rad(abs(degree)))
                plus = magnitude[row - 1, col] * (1 - t) + magnitude[row - 1, col + 1] * t
                minus = magnitude[row + 1, col - 1] * t + magnitude[row + 1, col] * (1 - t)
            else:
                print(row, col, 'error!  degree :', degree)

            max_value = max(plus, minus, magnitude[row, col])
            if max_value == magnitude[row, col]:
                temp = 1
            else:
                temp = 0
            largest_magnitude[row, col] = temp * magnitude[row, col]

    return largest_magnitude


def double_thresholding(src):
    dst = src.copy()

    # dst 범위 조정 0 ~ 255
    dst = (dst - np.min(dst)) / (np.max(dst) - np.min(dst))
    dst *= 255
    dst = dst.astype('uint8')

    # threshold는 정해진 값을 사용
    high_threshold_value = 40
    low_threshold_value = 5

    print(high_threshold_value, low_threshold_value)

    #####################################################
    # TODO                                              #
    # Double thresholding 수행                           #
    #####################################################
    dx = [0, 0, 1, 1, 1, -1, -1, -1]
    dy = [-1, 1, -1, 0, 1, -1, 0, 1]
    (h, w) = dst.shape
    q = []
    for row in range(h):
        for col in range(w):
            now = dst[row, col]
            if now >= high_threshold_value:
                dst[row, col] = 255
            elif now < low_threshold_value:
                dst[row, col] = 0
            else:
                q.append((row, col))

    for x,y in q:
        if dst[x,y] == 255 or dst[x,y] == 0:
            continue

        repeat = False
        for i in range(8):
            nx, ny = x + dx[i], y + dy[i]

            if repeat:
                break
            if dst[nx, ny] == 255:
                tmp_q = [(x, y)]
                dst[(x, y)] = 255

                while tmp_q:
                    now_x, now_y = tmp_q.pop()
                    for i in range(8):
                        nx = now_x + dx[i]
                        ny = now_y + dy[i]

                        if low_threshold_value <= dst[nx, ny] <= high_threshold_value:
                            tmp_q.append((nx, ny))
                            dst[nx, ny] = 255

    for row in range(h):
        for col in range(w):
            if low_threshold_value <= dst[row, col] <= high_threshold_value:
                dst[row,col] = 0

    # return dst
    dst = dst.astype('float32') / 255.0
    return dst


def canny_edge_detection(src):
    # Apply low pass filter
    I = apply_gaussian_filter(src, fsize=3, sigma=1)

    # Apply high pass filter
    Ix, Iy = apply_sobel_filter(I)

    # Get magnitude and angle
    magnitude = calc_magnitude(Ix, Iy)
    angle = calc_angle(Ix, Iy)

    # Apply non-maximum-supression
    after_nms = non_maximum_supression(magnitude, angle)

    # Apply double thresholding
    dst = double_thresholding(after_nms)

    return dst, after_nms, magnitude


if __name__ == '__main__':
    img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255.0

    canny, after_nms, magnitude = canny_edge_detection(img)

    # 시각화 하기 위해 0~1로 normalize (min-max scaling)
    magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))
    after_nms = (after_nms - np.min(after_nms)) / (np.max(after_nms) - np.min(after_nms))

    cv2.imshow('original', img)
    cv2.imshow('magnitude', magnitude)
    cv2.imshow('after_nms', after_nms)
    cv2.imshow('canny_edge', canny)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

