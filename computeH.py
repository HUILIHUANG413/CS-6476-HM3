import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
import normalizeHomogeneous
#H = computeH(t1, t2)
def  computeH(t1, t2):
    #Matrix L
    nPoints=t1.shape[1]
    row,col=t1.shape #2*N
    L=np.zeros((nPoints*2,9))
    #homogeneous coordinate
    t1_1=np.ones((row+1,col))
    t2_1=np.ones((row+1,col))
    t1_1[:-1,:]=t1
    t2_1[:-1,:]=t2
    #construct matrix L
    for i in range (0,nPoints):
        for j in range(0,2):
            if j==0:
                part1=t1_1[:,i].T
                part2=np.zeros(3)
                part3=-1*np.dot(t2_1[0,i],t1_1[:,i].T)
                L_tmp=np.zeros(6)
                L_tmp=np.append(part1,part2)
                L[2 * i + j, :] = np.append(L_tmp,part3)
               # L[2*i+j,:]=np.append(np.array(t1_1)[:,i].T,[0,0,0],-1*np.dot(t2_1[0,i],t1_1[:,i].T))
            if j==1:
                part2 = (np.array(t1_1)[:, i]).T
                part1 = np.zeros(3)
                part3 = -1 * np.dot(t2_1[1, i] , t1_1[:, i].T)
                L_tmp = np.zeros(6)
                L_tmp = np.append(part1, part2)
                L[2 * i + j, :] = np.append(L_tmp, part3)
               # L[2 * i + j, :] = np.append([0,0,0],(np.array(t1_1)[:, i]).T,  -1 * np.dot(t2_1[1, i] , t1_1[:, i].T))
    #w=eig value,v=eig vector
    w,v=np.linalg.eig(np.dot(L.T,L))
    min_i=np.argmin(w)
    #normalize
    #H=v[:,min_i]
    #H=H/np.sqrt(np.sum(H ** 2))
    H=v[:,min_i].reshape(3,3)

    return H

def main():
#for wdc
    #t1=np.load("points1.npy")
    #t2=np.load("points2.npy")
    #print(t1,"\n",t2)
#for crops
    t1=np.load("cc1.npy")
    t2=np.load("cc2.npy")
#for keble
    #t1=np.load("points1_keble_1.npy")
    #t2=np.load("points2_keble_2.npy")
    #t1=t1.T
    #t2=t2.T
    row, col = t1.shape  # 2*N
    # homogeneous coordinate
    t1_1 = np.ones((row + 1, col))
    t2_1 = np.ones((row + 1, col))
    t1_1[:-1, :] = t1
    t2_1[:-1, :] = t2
    H=computeH(t1,t2)
    t1_result=np.dot(H,t1_1)
    t1_result=normalizeHomogeneous.normalizeHomogeneous(t1_result)
    im2=imread("crop2.jpg")
    plt.imshow(im2)
    plt.title("Pairs in wdc")
    plt.scatter(t1_result[0,:],t1_result[1,:],s=1,c='b')
    #plt.figure()
    #im1=imread("wdc1.jpg")
    #plt.imshow(im1)
    plt.scatter(t2[0],t2[1],s=1,c='r')
    #plt.savefig("pairs_keble.png")
    plt.show()

    print(H)
if __name__ == '__main__':
    main()

