import numpy as np
import normalizeHomogeneous
import computeH
from imageio import imread
import matplotlib.pyplot as plt

#[warpIm,mergeIm] = warpImage(inputIm, refIm, H)
def warpImage(inputIm, refIm, H) :
    hInput,wInput,_=inputIm.shape
    hRef,wRef,_=refIm.shape
    corners=np.array([[1,1,wInput,wInput],[1,hInput,1,hInput],[1,1,1,1]])
    cornersWarped=np.dot(H,corners)
    cornersWarped=normalizeHomogeneous.normalizeHomogeneous(cornersWarped)
    x_min=np.min(cornersWarped[0,:])
    x_max=np.max(cornersWarped[0,:])
    y_min = np.min(cornersWarped[1, :])
    y_max = np.max(cornersWarped[1, :])
    width=np.int(np.ceil(x_max-x_min))
    hight=np.int(np.ceil(y_max-y_min))
    warpIm=np.zeros((hight*width, 3))
    Hinv=np.linalg.inv(H)

    xx=np.arange(x_min,x_max)
    yy=np.arange(y_min,y_max)
    [X,Y]=np.meshgrid(xx,yy)
    tmp=np.ones((1,X.shape[0]*X.shape[1]))
    #pointsProjected=np.vstack(X.T)
    #tmp1=np.append(X.T,Y.T)
    #pointsProjected=np.append(tmp1,tmp)
    tmpX=X.reshape(1,-1)
    pointsProjected=np.vstack((X.reshape(1,-1),Y.reshape(1,-1),tmp))
    pointsSorce=np.dot(Hinv,pointsProjected)
    pointsSorce=normalizeHomogeneous.normalizeHomogeneous(pointsSorce)

    xr=pointsSorce[0,:]
    xr=xr.astype(np.int)
    yr=pointsSorce[1,:]
    yr=yr.astype(np.int)

    for i in range(0,width*hight):
        #print("1")
        if (xr[i]>=1 and yr[i]>=1 and xr[i]<wInput and yr[i]<hInput):
            for c in range(0,3):
                warpIm[i,c]=inputIm[yr[i],xr[i],c]

    warpIm=np.reshape(warpIm,(hight,width,3))

    offsetX=1
    offsetY=1

    if(y_min<1):
        offsetY=np.int(round(np.abs(y_min)))
    if(x_min<1):
        offsetX=np.int(round(np.abs(x_min)))

    canvas=np.zeros((np.int(hRef+offsetY),np.int(wRef+offsetX),3))
    canvas=canvas.astype(np.int32)
    #print( canvas[offsetY:,offsetX:,:].shape)
    canvas[offsetY:,offsetX:,:]=refIm
    plt.figure()
    plt.imshow(canvas)
    #C=np.dstack((warpIm,canvas*1.8))
    #cv2.imshow("imfuse",C)
    #mergeIm=Image.blend(warpIm,canvas*1.8,alpha=0.5)
    #mergeIm=pymg.imfuse(warpIm,canvas*1.8)
    mergeIm=warpIm.copy()
    r,c,_=canvas.shape
    for i in range(0,r):
        for j in range(0,c):
            if (i<hight and j<width):
                if(canvas[i,j,1]!=0 and canvas[i,j,2]!=0 and canvas[i,j,0]!=0):
                    mergeIm[i,j,:]=canvas[i,j,:]
    #mergeIm = np.array(mergeIm, np.int32)
    #plt.imshow(mergeIm)

    return warpIm,mergeIm

def main():
    t1=np.load("points1_bill.npy")
    t2=np.load("points2_bill.npy")
    H=computeH.computeH(t1,t2)
    inputim = imread("billboard.jpg")
    refim = imread("amazon.jpg")
    warpIm,mergIm=warpImage(inputim,refim,H)
    #warp
    plt.figure()
    plt.title("warpImage of billbord")
    warpIm = np.array(warpIm, np.int32)
    plt.imshow(warpIm)
    plt.savefig("Warpimage_billbord_1.png")
    #merge

    mergIm = np.array(mergIm, np.int32)
    plt.figure()
    plt.title("mergeImage ")
    plt.imshow(mergIm)
    plt.savefig("Mergeimage_billbord_1.png")
    plt.show()


if __name__ == '__main__':
    main()

