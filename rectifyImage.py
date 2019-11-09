import numpy as np
import normalizeHomogeneous
import computeH
from imageio import imread
import matplotlib.pyplot as plt
import math
def rectifyImage(inputIm, corners,H):
    input_x = inputIm.shape[0]
    input_y = inputIm.shape[1]
    #ref_x = refIm.shape[0]
    #ref_y = refIm.shape[1]
    #corners = np.array([[0, input_y - 1, 0, input_y - 1], [0, 0, input_x - 1, input_x - 1], [1, 1, 1, 1]])
    cornersWarped = np.dot(H, corners)
    cornersWarped = normalizeHomogeneous.normalizeHomogeneous(cornersWarped)
    x_min = min(cornersWarped[0, :])
    x_max = max(cornersWarped[0, :])
    y_min = min(cornersWarped[1, :])
    y_max = max(cornersWarped[1, :])
    print(x_min, y_min, x_max, y_max)
    # boundary points
    point1 = min(x_min, 0)
    point2 =x_max
    point3 = min(y_min, 0)
    point4 =y_max
    Hinv = np.linalg.inv(H)
    # initial
    width = np.int(math.ceil(x_max - point1) - 1)
    height = np.int(math.ceil(y_max - point3) - 1)
    warpIm = np.zeros((height, width, 3))
    mergeIm = warpIm.copy()
    row = []
    col = []
    k = 0
    for i in range(np.int(point1), np.int(point2)):
        row.append(i)
    for j in range(np.int(point3), np.int(point4)):
        col.append(j)
    X, Y = np.meshgrid(row, col)
    box = np.column_stack((X.ravel(), Y.ravel()))
    result_box = (np.append(box, np.ones((len(box), 1)), axis=1)).T
    print(result_box)
    for i in range(np.int(point3), np.int(point4)):
        for j in range(np.int(point1), np.int(point2)):
            if (len(result_box.T) > k):
                tmp = result_box.T[k]
                inv = np.matmul(Hinv, tmp)
                input_col = inv[0] / inv[2]
                input_row = inv[1] / inv[2]
                k = k + 1
            if (0 < input_row and 0 < input_col and input_row < input_x - 1 and input_col < input_y - 1):
                r = np.int(input_row)
                c = np.int(input_col)
                tmp1 = input_row - r
                tmp2 = input_col - c
                warpIm_r = np.int(i - point3)
                warpIm_c = np.int(j - point1)
                mergeIm_r = np.int(i - point3)
                mergeIm_c = np.int(j - point1)
                # mapping warp
                warpIm[warpIm_r, warpIm_c] = (1 - tmp1) * (1 - tmp2) * inputIm[r][c] + tmp1 * (1 - tmp2) * inputIm[r][
                    c + 1] \
                                             + tmp1 * tmp2 * inputIm[r + 1][c + 1] + (1 - tmp1) * tmp2 * inputIm[r + 1][
                                                 c]
                # mapping merge
               # mergeIm[mergeIm_r, mergeIm_c] = (1 - tmp1) * (1 - tmp2) * inputIm[r][c] + tmp1 * (1 - tmp2) * \
               #                                 inputIm[r][c + 1] \
               #                                 + tmp1 * tmp2 * inputIm[r + 1][c + 1] + (1 - tmp1) * tmp2 * \
               #                                 inputIm[r + 1][c]

    return warpIm

def main():
    inputim = imread("tiananmen.jpg")
    plt.figure()
    plt.imshow(inputim)
    row,col,_=inputim.shape
    t1=np.load("tiananmen.npy")
    maxX=np.max(t1[0,:])
    maxY=np.max(t1[1,:])
    minX=np.min(t1[0,:])
    minY=np.min(t1[1,:])
    t2=np.array([[0,maxX-minX,0,maxX-minX],[0,0,maxY-minY,maxY-minY]])
    row, col = t1.shape
    t1_1 = np.ones((row + 1, col))
    t2_1 = np.ones((row + 1, col))
    t1_1[:-1, :] = t1
    t2_1[:-1, :] = t2
    H=computeH.computeH(t1,t2)
    warped=np.dot(H,t1_1)
    warped=normalizeHomogeneous.normalizeHomogeneous(warped)
    rectifyIm=rectifyImage(inputim,t1_1,H)
    print(H)

    plt.figure()
    plt.title("rectify image_tiananmen")
    rectifyIm = np.array(rectifyIm, np.int32)
    plt.imshow(rectifyIm)
    plt.savefig("rectify_tiananmen.png")
    plt.show()




if __name__ == '__main__':
    main()