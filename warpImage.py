import numpy as np
import normalizeHomogeneous
import computeH
from imageio import imread
import matplotlib.pyplot as plt
import math


# [warpIm,mergeIm] = warpImage(inputIm, refIm, H)
def warpImage(inputIm, refIm, H):
    input_x=inputIm.shape[0]
    input_y=inputIm.shape[1]
    ref_x=refIm.shape[0]
    ref_y=refIm.shape[1]
    corners = np.array([[0, input_y-1, 0, input_y-1], [0, 0, input_x-1, input_x-1], [1, 1, 1, 1]])
    cornersWarped = np.dot(H, corners)
    cornersWarped = normalizeHomogeneous.normalizeHomogeneous(cornersWarped)
    x_min = min(cornersWarped[0, :])
    x_max = max(cornersWarped[0, :])
    y_min = min(cornersWarped[1, :])
    y_max = max(cornersWarped[1, :])
    print(x_min, y_min, x_max, y_max)
    #boundary points
    point1=min(x_min,0)
    point2=max(x_max,ref_y)
    point3=min(y_min,0)
    point4=max(y_max,ref_x)
    Hinv = np.linalg.inv(H)
    #initial
    width = np.int(math.ceil(point2 - point1) - 1)
    height = np.int(math.ceil(point4 - point3) - 1)
    warpIm = np.zeros((height, width, 3))
    mergeIm = warpIm.copy()
    row=[]
    col=[]
    k=0
    for i in range(np.int(point1),np.int(point2)):
        row.append(i)
    for j in range(np.int(point3),np.int(point4)):
        col.append(j)
    X,Y=np.meshgrid(row,col)
    box = np.column_stack((X.ravel(), Y.ravel()))
    result_box= (np.append(box, np.ones((len(box), 1)), axis=1)).T
    print(result_box)
    mergeIm = warpIm.copy()
    for i in range(np.int(point3),np.int(point4)):
        for j in range (np.int(point1),np.int(point2)):
            if (len(result_box.T)>k):
                tmp=result_box.T[k]
                inv=np.matmul(Hinv,tmp)
                input_col=inv[0]/inv[2]
                input_row=inv[1]/inv[2]
                k=k+1
            if (0<input_row and 0<input_col and input_row<input_x-1 and input_col< input_y-1):
                r=np.int(input_row)
                c=np.int(input_col)
                tmp1=input_row-r
                tmp2=input_col-c
                warpIm_r=np.int(i-point3)
                warpIm_c=np.int(j-point1)
                mergeIm_r=np.int(i-point3)
                mergeIm_c=np.int(j-point1)
                #mapping warp
                warpIm[warpIm_r,warpIm_c]=(1-tmp1)*(1-tmp2)*inputIm[r][c]+tmp1*(1-tmp2)*inputIm[r][c+1]\
                +tmp1*tmp2*inputIm[r+1][c+1]+(1-tmp1)*tmp2*inputIm[r+1][c]
                #mapping merge
                mergeIm[mergeIm_r,mergeIm_c] = (1-tmp1)*(1-tmp2)*inputIm[r][c]+tmp1*(1-tmp2)*inputIm[r][c+1]\
                +tmp1*tmp2*inputIm[r+1][c+1]+(1-tmp1)*tmp2*inputIm[r+1][c]
    #merge the ref to warp
    for i in range(0,result_box.shape[1]):
        r=result_box[0,i]
        c=result_box[1,i]
        if (r>0 and c > 0 and r < ref_y-1 and c < ref_x-1):
            mergeIm_r = np.int(c - point3)
            mergeIm_c = np.int(r - point1)
            mergeIm[mergeIm_r,mergeIm_c,:]=refIm[np.int(c),np.int(r),:]
    return warpIm, mergeIm


def main():
#points for crop
    #t1=np.load("cc1.npy")
    #t2=np.load("cc2.npy")
    #t1=np.array(t1.T)
    #t2=np.array(t2.T)
#points for wdc
    t1 = np.load("points1.npy")
    t2 = np.load("points2.npy")
#points for question 4
    #t1 = np.load("points1_keble.npy")
    #t2 = np.load("points2_keble.npy")
#points for question 5
    #t1 = np.load("points1_bill.npy")
    #t2 = np.load("points2_bill.npy")
    # print(t1,"\n",t2)
    H = computeH.computeH(t1, t2)
    # print(H)
#images for crops
    #inputim=imread("crop1.jpg")
    #refim=imread("crop2.jpg")
#images for wdc
    inputim = imread("wdc1.jpg")
    refim = imread("wdc2.jpg")
#images for question 4
    #inputim=imread("keble_a.jpg")
    #refim=imread("keble_b.jpg")
#image for question 5
    #inputim = imread("billboard.jpg")
    #refim = imread("amazon.jpg")
    warpIm, mergIm = warpImage(inputim, refim, H)
    # warp
    plt.figure()
    plt.title("warpImage of wdc")
    warpIm = np.array(warpIm, np.int32)
    plt.imshow(warpIm)
    #plt.savefig("Warpimage_wdc.png")
    # merge
    mergIm = np.array(mergIm, np.int32)
    plt.figure()
    plt.title("mergeImage of wdc")
    plt.imshow(mergIm)
    #plt.savefig("Mergeimage_wdc.png")
    plt.show()


if __name__ == '__main__':
    main()

