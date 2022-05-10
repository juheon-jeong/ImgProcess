def my_bilinear(src, scale):
    #########################
    # TODO                  #
    # my_bilinear 완성      #
    #########################
    (h, w) = src.shape
    h_dst = int(h * scale + 0.5)
    w_dst = int(w * scale + 0.5)

    dst = np.zeros((h_dst, w_dst))

    # bilinear interpolation 적용
    for row in range(h_dst):
        for col in range(w_dst):
            # 참고로 꼭 한줄로 구현해야 하는건 아닙니다 여러줄로 하셔도 상관없습니다.(저도 엄청길게 구현했습니다.)

            y = min(int(row/scale), h - 2)
            x = min(int(col/scale), w - 2)

            fx1 = (col/scale) - int(col/scale)
            fx2 = 1 - fx1
            fy1 = (row/scale) - int(row/scale)
            fy2 = 1 - fy1

            dst[row,col] = src[y,x]*fx2*fy2 + src[y,x+1]*fx1*fy2 + src[y+1,x]*fx2*fy1 + src[y+1,x+1]*fx1*fy1

    return dst