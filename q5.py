import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QInputDialog, QLineEdit
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import optimizers
import numpy as np
# from tensorflow.keras.layers.core import Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
import random
label_class=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def ShowHyper():
    print('hyperparameters:')
    print('batch size: ', 32)
    print('learning rate: ',0.001)
    print('optimizer: ','SGD')
    return

def ShowImg():
    plt.figure()
    for i in range(1,11):
        index = random.randint(0,49999)
        plt.subplot(2,5,i)
        plt.title(label_class[y[index][0]])
        plt.imshow(x[index])
    plt.show()
    return

def ModelStructure():
    weight_decay = 0.0005
    model = keras.Sequential()
    model.add(Conv2D(64, (3, 3), padding='same',
    input_shape=(32,32,3),kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    #layer2 32*32*64
    model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #layer3 16*16*64
    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    #layer4 16*16*128
    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #layer5 8*8*128
    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    #layer6 8*8*256
    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    #layer7 8*8*256
    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #layer8 4*4*256
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    #layer9 4*4*512
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    #layer10 4*4*512
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #layer11 2*2*512
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    #layer12 2*2*512
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    #layer13 2*2*512
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    #layer14 1*1*512
    model.add(Flatten())
    model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #layer15 512
    model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #layer16 512
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.summary()

def ShowAccuracy():
    acc = cv2.imread('accuracy.png')
    loss = cv2.imread('loss.png')
    cv2.imshow('acc', acc)
    cv2.imshow('loss', loss)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def TestImg():
    new_model = keras.models.load_model('my_model')
    testIndex = int(testInput.text())
    img = x_test[testIndex].reshape((1,32,32,3))
    ans = new_model.predict(img)[0]
    bars = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # print(ans)
    plt.figure(1)
    plt.imshow(x_test[testIndex])
    plt.figure(2)
    plt.bar(bars, ans)
    plt.xticks(range(len(bars)), bars, rotation=45)
    plt.show()

def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

app = QApplication([])
app.setApplicationName('2020 Opencvdl HW1')
app.setStyle('Fusion')
window = QWidget()
window.setGeometry(500,100,200,400)
solve_cudnn_error()
(x,y),(x_test,y_test) = datasets.cifar10.load_data()

btn1 = QPushButton(window)
btn1.setGeometry(10,10,150,30)
btn1.setText('1.Show Train Images')
btn1.clicked.connect(lambda : ShowImg())

btn2 = QPushButton(window)
btn2.setGeometry(10,70,150,30)
btn2.setText('2.Show Hyperparameters')
btn2.clicked.connect(lambda : ShowHyper())

btn3 = QPushButton(window)
btn3.setGeometry(10,130,150,30)
btn3.setText('3.Show Model Structure')
btn3.clicked.connect(lambda : ModelStructure())

btn4 = QPushButton(window)
btn4.setGeometry(10,190,150,30)
btn4.setText('4.Show Accuracy')
btn4.clicked.connect(lambda : ShowAccuracy())

testInput = QLineEdit(window)
testInput.setGeometry(10,250,150,30)

btn5 = QPushButton(window)
btn5.setGeometry(10,310,150,30)
btn5.setText('5.Test')
btn5.clicked.connect(lambda : TestImg())



window.show()
app.exec_()