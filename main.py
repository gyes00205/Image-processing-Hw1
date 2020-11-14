from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QInputDialog, QLineEdit
import cv2
import matplotlib.pyplot as plt
import numpy as np

mergeImg = 0

def LoadImage():
    loadImg = cv2.imread("./Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg")
    cv2.imshow("1",loadImg)
    print("Height = ", loadImg.shape[0])
    print("Width = ", loadImg.shape[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ColorSeparation():
    colorImg = cv2.imread("./Dataset_opencvdl/Q1_Image/Flower.jpg")
    b,g,r = cv2.split(colorImg)
    z = np.zeros_like(b)
    cv2.imshow("1", colorImg)
    cv2.imshow("2", cv2.merge((b,z,z)))
    cv2.imshow("3", cv2.merge((z,g,z)))
    cv2.imshow("4", cv2.merge((z,z,r)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ImageFlip():
    originImg = cv2.imread("./Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg")
    flipImg = cv2.flip(originImg, 1)
    cv2.imshow('Original Image', originImg)
    cv2.imshow('Result', flipImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def UpdateBlend(X):
    global mergeImg
    originImg = cv2.imread("./Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg")
    flipImg = cv2.flip(originImg, 1)
    p1 = 1.0 - cv2.getTrackbarPos('BLEND', 'BLENDING')/255
    p2 = 1.0 - p1
    mergeImg = cv2.addWeighted(originImg,p1,flipImg,p2,0)  
    cv2.imshow('BLENDING', mergeImg)
    
    

def Blending():
    cv2.namedWindow('BLENDING')
    cv2.createTrackbar('BLEND', 'BLENDING', 0, 255, UpdateBlend)
    cv2.setTrackbarPos('BLEND', 'BLENDING', 120)
    

def MedianFilter():
    originImg = cv2.imread("./Dataset_opencvdl/Q2_Image/Cat.png")
    medianImg = cv2.medianBlur(originImg, 7)
    cv2.imshow("Original", originImg)
    cv2.imshow("median", medianImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def GaussianBlurCall():
    originImg = cv2.imread("./Dataset_opencvdl/Q2_Image/Cat.png")
    gaussianImg = cv2.GaussianBlur(originImg, (3, 3), 0)
    cv2.imshow("Original", originImg)
    cv2.imshow("Gaussian", gaussianImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def BilateralFilter():
    originImg = cv2.imread("./Dataset_opencvdl/Q2_Image/Cat.png")
    bilateralImg = cv2.bilateralFilter(originImg, 9, 90, 90)
    cv2.imshow("Original", originImg)
    cv2.imshow("Bilateral", bilateralImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def GaussianBlurSelf():
    Gnorm = np.array([[0.045,0.122,0.045],
                      [0.122,0.332,0.122],
                      [0.045,0.122,0.045]])
    originImg = cv2.imread("./Dataset_opencvdl/Q3_Image/Chihiro.jpg", 0)
    height, width = originImg.shape 
    gaussianImg = np.zeros_like(originImg)
    for x in range(1,width-1):
        for y in range(1,height-1):
            val = sum(sum(Gnorm*originImg[y-1:y+2,x-1:x+2]))
            gaussianImg[y][x] = val
    cv2.imshow("Gaussian", gaussianImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def SobelX():
    Max = 255
    Min = 0
    Gnorm = np.array([[0.045,0.122,0.045],
                      [0.122,0.332,0.122],
                      [0.045,0.122,0.045]])
                      
    gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    
    originImg = cv2.imread("./Dataset_opencvdl/Q3_Image/Chihiro.jpg", 0)
    height, width = originImg.shape 
    sobelXImg = np.zeros_like(originImg)
    gaussianImg = np.zeros_like(originImg)
    for x in range(1,width-1):
        for y in range(1,height-1):
            val = sum(sum(Gnorm*originImg[y-1:y+2,x-1:x+2]))
            gaussianImg[y][x] = val

    for x in range(1,width-1):
        for y in range(1,height-1):
            val = sum(sum(gx*gaussianImg[y-1:y+2,x-1:x+2]))
            if(val>Max):
                val = Max
            if(val<Min):
                val = Min
            sobelXImg[y][x] = val
            
    
    # print(Max, Min)
    # sobelXImg = sobelXImg/8
    # print(sobelXImg)
    cv2.imshow("SobelX", sobelXImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def SobelY():
    Max = 255
    Min = 0
    Gnorm = np.array([[0.045,0.122,0.045],
                      [0.122,0.332,0.122],
                      [0.045,0.122,0.045]])
    gy = np.array([[-1,-2,-1],
                   [ 0, 0, 0],
                   [ 1, 2, 1]])
    
    originImg = cv2.imread("./Dataset_opencvdl/Q3_Image/Chihiro.jpg", 0)
    height, width = originImg.shape 
    sobelYImg = np.zeros_like(originImg)
    gaussianImg = np.zeros_like(originImg)
    for x in range(1,width-1):
        for y in range(1,height-1):
            val = sum(sum(Gnorm*originImg[y-1:y+2,x-1:x+2]))
            gaussianImg[y][x] = val

    for x in range(1,width-1):
        for y in range(1,height-1):
            val = sum(sum(gy*gaussianImg[y-1:y+2,x-1:x+2]))
            if(val>Max):
                val = Max
            if(val<Min):
                val = Min
            sobelYImg[y][x] = val
    cv2.imshow("SobelY", sobelYImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def MagnitudeSobel():
    Gnorm = np.array([[0.045,0.122,0.045],
                      [0.122,0.332,0.122],
                      [0.045,0.122,0.045]])
    gy = np.array([[-1,-2,-1],
                   [ 0, 0, 0],
                   [ 1, 2, 1]])
    gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    
    originImg = cv2.imread("./Dataset_opencvdl/Q3_Image/Chihiro.jpg", 0)
    height, width = originImg.shape 
    magnitudeImg = np.zeros_like(originImg)
    gaussianImg = np.zeros_like(originImg)
    for x in range(1,width-1):
        for y in range(1,height-1):
            val = sum(sum(Gnorm*originImg[y-1:y+2,x-1:x+2]))
            gaussianImg[y][x] = val

    for x in range(1,width-1):
        for y in range(1,height-1):
            dx = sum(sum(gx*originImg[y-1:y+2,x-1:x+2]))
            dy = sum(sum(gy*originImg[y-1:y+2,x-1:x+2]))
            magnitudeImg[y][x] = (dx**2 + dy**2)**0.5
    cv2.imshow("Magnitude", magnitudeImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def TransformationSelf():
    deg = int(degInput.text())
    scaling = float(scalingInput.text())
    tx = int(txInput.text())
    ty = int(tyInput.text())
    # print(deg,scaling,tx,ty)
    originImg = cv2.imread("./Dataset_opencvdl/Q4_Image/Parrot.png")
    row, col = originImg.shape[0], originImg.shape[1]
    # M = cv2.getRotationMatrix2D((160,84), deg, scaling)
    # M[0,2], M[1,2] = tx, ty
    # rstImg = cv2.warpAffine(originImg, M, (col,row))
    H = np.float32([[1,0,tx], [0,1,ty]])
    M = cv2.getRotationMatrix2D(((160+tx),(84+ty)), deg, scaling)
    rstImg = cv2.warpAffine(originImg, H, (col,row))
    rstImg = cv2.warpAffine(rstImg, M, (col,row))
    # cv2.circle(rstImg, (360,384), 5, (0, 255, 255), -1)
    # cv2.circle(originImg, (160,84), 5, (0, 255, 255), -1)
    # cv2.circle(rstImg, (160,84), 5, (0, 255, 255), -1)
    cv2.imshow('Original Image', originImg)
    cv2.imshow('Image RST', rstImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


app = QApplication([])
app.setApplicationName('2020 Opencvdl HW1')
app.setStyle('Fusion')
window = QWidget()
window.setGeometry(500,100,800,500)

label1 = QLabel(window)
label1.setGeometry(10,10,110,60)
label1.setText('1.Image Processing')

btn11 = QPushButton(window)
btn11.setGeometry(10,70,150,30)
btn11.setText('1.1 Load Image')
btn11.clicked.connect(lambda : LoadImage())

btn12 = QPushButton(window)
btn12.setGeometry(10,130,150,30)
btn12.setText('1.2 Color seperation')
btn12.clicked.connect(lambda : ColorSeparation())

btn13 = QPushButton(window)
btn13.setGeometry(10,190,150,30)
btn13.setText('1.3 Image Flipping')
btn13.clicked.connect(lambda : ImageFlip())

btn14 = QPushButton(window)
btn14.setGeometry(10,250,150,30)
btn14.setText('1.4 Blending')
btn14.clicked.connect(lambda : Blending())

label2 = QLabel(window)
label2.setGeometry(180,10,110,60)
label2.setText('2.Image Smoothing')

btn21 = QPushButton(window)
btn21.setGeometry(180,70,150,30)
btn21.setText('2.1 Median Filter')
btn21.clicked.connect(lambda : MedianFilter())

btn22 = QPushButton(window)
btn22.setGeometry(180,130,150,30)
btn22.setText('2.2 Gaussian Blur')
btn22.clicked.connect(lambda : GaussianBlurCall())

btn23 = QPushButton(window)
btn23.setGeometry(180,190,150,30)
btn23.setText('2.3 Bilateral Filter')
btn23.clicked.connect(lambda : BilateralFilter())

label3 = QLabel(window)
label3.setGeometry(350,10,110,60)
label3.setText('3.Edge Detection')

btn31 = QPushButton(window)
btn31.setGeometry(350,70,150,30)
btn31.setText('3.1 Gaussian Blur')
btn31.clicked.connect(lambda : GaussianBlurSelf())

btn32 = QPushButton(window)
btn32.setGeometry(350,130,150,30)
btn32.setText('3.2 Sobel X')
btn32.clicked.connect(lambda : SobelX())

btn33 = QPushButton(window)
btn33.setGeometry(350,190,150,30)
btn33.setText('3.3 Sobel Y')
btn33.clicked.connect(lambda : SobelY())

btn34 = QPushButton(window)
btn34.setGeometry(350,250,150,30)
btn34.setText('3.4 Magnitude')
btn34.clicked.connect(lambda : MagnitudeSobel())

label4 = QLabel(window)
label4.setGeometry(520,10,110,60)
label4.setText('4.Transformation')

rotationLabel = QLabel(window)
rotationLabel.setGeometry(520,70,70,30)
rotationLabel.setText('Rotation:')
degInput = QLineEdit(window)
degInput.setGeometry(580,70,150,30)
degLabel = QLabel(window)
degLabel.setGeometry(735,70,50,30)
degLabel.setText('deg')

scalingLabel = QLabel(window)
scalingLabel.setGeometry(520,130,70,30)
scalingLabel.setText('Scaling:')
scalingInput = QLineEdit(window)
scalingInput.setGeometry(580,130,150,30)

txLabel = QLabel(window)
txLabel.setGeometry(520,190,70,30)
txLabel.setText('Tx:')
txInput = QLineEdit(window)
txInput.setGeometry(580,190,150,30)
pixelxLabel = QLabel(window)
pixelxLabel.setGeometry(735,190,50,30)
pixelxLabel.setText('pixel')

tyLabel = QLabel(window)
tyLabel.setGeometry(520,250,70,30)
tyLabel.setText('Ty:')
tyInput = QLineEdit(window)
tyInput.setGeometry(580,250,150,30)
pixelyLabel = QLabel(window)
pixelyLabel.setGeometry(735,250,50,30)
pixelyLabel.setText('pixel')

transformbtn = QPushButton(window)
transformbtn.setGeometry(520,310,240,30)
transformbtn.setText('4. Transformation')
transformbtn.clicked.connect(lambda : TransformationSelf())
# x= np.array([[1, 0, -1],
#              [2, 0, -2],
#              [1, 0, -1]])
# y= np.array([[1, 0, -1],
#              [2, 0, -2],
#              [1, 0, -1]])

# print(x+1)
# print(sum(x*y))
# print(sum(sum(x*y)))

window.show()
app.exec_()