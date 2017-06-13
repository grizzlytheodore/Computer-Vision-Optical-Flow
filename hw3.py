import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal

def rescale(matrix):
    scaled = ((matrix - matrix.min()) * (255 / (matrix.max() - matrix.min()))).astype(np.uint8)
    return scaled

def computeC2(sigma, theta):
    numerater = 0
    denominator = 0
    filterRadius = sigma*3
    e = [np.cos(theta), np.sin(theta)]
    num = (np.pi/(2*sigma))
    for row in range(-filterRadius, filterRadius+1):
        for col in range(-filterRadius, filterRadius+1):
            u = [row,col]
            u2 = (row**2) + (col**2)
            denominator += np.exp(-u2/(2*(sigma**2)))
            numerater += complex(np.cos(num*np.dot(u, e)),np.sin(num*np.dot(u,e)))*np.exp(-u2/(2*(sigma**2)))
    c2 = numerater / denominator
    return c2

def computeC1(sigma, theta, c2):
    filterRadius = sigma*3
    z=0
    e = [np.cos(theta), np.sin(theta)]
    num = np.pi / (2*sigma)
    for row in range(-filterRadius, filterRadius+1):
        for col in range(-filterRadius, filterRadius+1):
            u = [row,col]
            u2 = (row**2)+(col**2)
            z += ((1 - 2 * c2 * np.cos(num*np.dot(u,e))+(c2**2))*np.exp(-u2/(sigma**2)))
    c1 = sigma/(z**0.5)
    return c1

def psiFunc(row,col,c1,c2,sigma,theta):
    num = np.pi/(2*sigma)
    u2 = (row**2)+(col**2)
    u = [row,col]
    e = [np.cos(theta), np.sin(theta)]
    psi = (c1/sigma) * (complex(np.cos(num*np.dot(u,e)), np.sin(num*np.dot(u,e))) - c2)* np.exp(-u2/(2*(sigma**2)))
    return psi

def makeWavelet(sigma, theta, morletImaginary):
    filterRadius = sigma * 3
    c2 = computeC2(sigma, theta)
    c1 = computeC1(sigma, theta, c2)
    for row in range(-filterRadius, filterRadius+ 1):
        for col in range(-filterRadius, filterRadius+ 1):
            morlet = psiFunc(row*1., col*1., c1, c2, sigma, theta)
            #morletReal[x+filterRadius][y+filterRadius] = morlet.real
            morletImaginary[row+filterRadius][col+filterRadius] = morlet.imag
    return

def makeWaveletList(sigma, Theta):
    filterRadius = sigma* 3
    iList = []
    #morletReal = np.zeros((filterRadius*2+1,filterRadius*2+1))
    morletImaginary = np.zeros((filterRadius*2+1,filterRadius*2+1))
    for theta in Theta:
        makeWavelet(sigma, theta, morletImaginary)
        #rList.append(np.matrix.copy(morletReal))
        iList.append(np.matrix.copy(morletImaginary))
    return iList

def convolveList(sigma, Theta, image_1):
    iList = makeWaveletList(sigma, Theta)
    cList = []
    for i in range (0,len(iList)):
        #left real
        convolved_image = signal.convolve2d(image_1, iList[i], mode='same')
        cList.append(np.matrix.copy(convolved_image))
    return cList

def plot3(m1, m2, m3, title):
    plt.suptitle(title)
    plt.subplot(1,3,1)
    plt.imshow(m1, cmap= 'gray')
    plt.subplot(1,3,2)
    plt.imshow(m2, cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(m3, cmap= 'gray')
    plt.show()
    return

def plot2(m1,m2, title):
    plt.suptitle(title)
    plt.subplot(1, 2, 1)
    plt.imshow(m1, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(m2, cmap='gray')
    plt.show()
    return

def computeW_ThetaMap(sigma, Theta, image_1):
    cList1 = convolveList(sigma, Theta, image_1)
    #cList2 = convolveList(sigma,Theta, image_2)
    [r,c] = cList1[0].shape
    for row in range (0, r):
        for col in range(0, c):
            for n in range (0, len(cList1)):
                if abs(W_im_max[row][col]) < abs(cList1[n][row][col]):
                    W_im_max[row][col] = cList1[n][row][col]
                    Theta_map[row][col] = n * np.pi/4
    return

def createGaussian(sigma):
    matrix = np.zeros([6*sigma+1,6*sigma+1])
    filterRadius = sigma * 3
    for row in range(-filterRadius, filterRadius+1):
        for col in range(-filterRadius, filterRadius+1):
            matrix[row+filterRadius][col+filterRadius] = (1/(2*np.pi*sigma**2)) * np.exp(-(row**2 + col**2)/(2*sigma**2))
    return matrix

def decimate(matrix, factor):
    return signal.decimate(signal.decimate(matrix, factor, axis=0, zero_phase=True), factor, zero_phase=True)

def test1(row,col, threshold=0.03):
    bool1 = False
    bool2 = False
    for i in range(-1,2):
        for j in range(-1, 2):
            if abs(W_im_max_d[row+i][col+j]) > threshold:
                bool1 = True
            if abs(time_derivative_d[row+i][col+j]) > threshold:
                bool2 = True
    return bool1 and bool2

def constructA(row, col, windowsize = 5):
    index = 0
    A = np.zeros([windowsize**2, 2])
    radius = windowsize /2
    for i in range (- radius, radius+1):
        for j in range (- radius, radius + 1):
            A[index][0] = W_im_max_d[row+i][col+j] * np.cos(Theta_map_d[row+i][col+j])
            A[index][1] = W_im_max_d[row+i][col+j] * np.sin(Theta_map_d[row+i][col+j])
            index += 1
    return A

def getTimeDiffWindow(row,col, windowsize = 5):
    time_diff = np.zeros([windowsize**2,1])
    index = 0
    radius = windowsize /2
    for i in range (- radius, radius+1):
        for j in range (- radius, radius + 1):
            time_diff[index][0] = time_derivative_d[row+i][col+j]
            index += 1
    return time_diff

def getVectorMatrix():
    [r,c] = W_im_max_d.shape
    matrixU = np.zeros([r,c])
    matrixV = np.zeros([r,c])
    for row in range(3,r-3):
        for col in range(3,c-3):
            if test1(row, col) == True:
                A = constructA(row, col)
                AT = np.transpose(A)
                ATA = np.dot(AT, A)
                w, v = np.linalg.eig(ATA)
                if min(w) >= 0.1:
                    [matrixU[row][col], matrixV[row][col]] = -np.dot(np.dot(np.linalg.inv(ATA), AT), getTimeDiffWindow(row, col))
                else:
                    [matrixU[row][col], matrixV[row][col]] = -time_derivative_d[row][col]/W_im_max_d[row][col] * [np.cos(Theta_map_d[row][col], np.sin(Theta_map_d[row][col]))]
    return [matrixU, matrixV]

def plotVectors(matrixU, matrixV):
    plt.figure()
    plt.title('Optical Flow Solution')
    [x,y] = matrixU.shape
    plt.quiver(-matrixU,matrixV)
    plt.show()
    return




#   CREATE BOXES
boxBefore = np.zeros([100, 100]) + 100.
boxAfter = np.zeros([100,100]) + 100.
for row in range (-15, 16):
    for col in range(-15, 16):
        boxBefore[50+row][50+col] = 200
        boxAfter[54+row][54+col] = 200
title = 'Box before moving & after moving'
#   PLOT BOXES
#plot2(boxBefore, boxAfter, title)

#   READ MINI COOPER
minibefore = cv2.imread('frame10.png',0)
miniafter = cv2.imread('frame11.png',0)
title = 'Mini Cooper'
#   PLOT MINI COOPER
#plot2(minibefore, miniafter,title )


#   INPUTS
image_1 = boxBefore
image_2 = boxAfter
sigma = 6
#image_1 = minibefore
#image_2 = miniafter
#sigma = 12
Theta = [0, np.pi/4, np.pi/2, np.pi*3/4]

#   CREATE W_IM_MIN, THETA_MAP
W_im_max = np.zeros(image_1.shape)
Theta_map = np.zeros(image_1.shape)
computeW_ThetaMap(sigma, Theta, image_1)
title = 'Intensity Edge Image, Angle Image'
#   PLOT W_IM_MAX, THETA_MAP
plot2(rescale(W_im_max), Theta_map, title)


#   CREATE GAUSSIAN
gaussian = createGaussian(sigma)
#   CONVOLVE BOXES W GAUSSIAN, TIME DERIVATIVE
convolvedBefore = signal.convolve2d(image_1, gaussian)
convolvedAfter = signal.convolve2d(image_2, gaussian)
time_derivative = convolvedBefore - convolvedAfter
title = 'Convolved Before, Convolved After'
#   PLOT CONVOLVED BOXES
#plot2(convolvedBefore, convolvedAfter, title)

#   CREATE X-EDGE Y-EDGE
x_edge = W_im_max * np.cos(Theta_map)
y_edge = W_im_max * np.sin(Theta_map)
title = "X-edge, Y-edge, Image difference"
#   PLOT X-EDGE, Y-EDGE, IMAGE DIFFERENCE
plot3(x_edge, y_edge, time_derivative, title)

#   DECIMATING
W_im_max_d = decimate(W_im_max, sigma)
Theta_map_d = decimate(Theta_map, sigma)
title1 = 'Decimated Edge Image, Angle Image'
time_derivative_d = decimate(time_derivative, sigma)
x_edge_d = decimate(x_edge, sigma)
y_edge_d = decimate(y_edge, sigma)
title2 = 'Decimated X Edge, Y Edge, Time Difference'
#   PLOT DECIMATED
plot2(W_im_max_d, Theta_map_d,title1)
plot3(x_edge_d, y_edge_d, time_derivative_d,title2)

#   A, AT, ATA
A = constructA(6,6)
AT = np.transpose(A)
ATA = np.dot(AT,A)
title = 'AT, A, ATA'
#   PLOT
#plot3(AT, A, ATA, title)

#   GET VECTOR MATRIX
[matrixU, matrixV] = getVectorMatrix()
plotVectors(matrixU, matrixV)
