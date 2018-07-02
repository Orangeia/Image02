import numpy as np
import cv2
import math
from scipy.ndimage.filters import convolve
from numpy import uint8, float32, float64, log, pi, sin, cos, abs, sqrt

w3 = np.ones((3,3)) / float(3*3)    #平滑化のためのマスク
w5 = np.ones((5,5)) / float(5*5)    
se = np.array([[-1, -1, -1],        #先鋭化のためのマスク
              [-1, 9, -1],
              [-1, -1, -1]])


def myfunc(i):
    pass # do nothing

def s_tone (x):  #S字カーブ
    y = (np.sin(np.pi * (x/255 - 0.5)) + 1)/2 * 255 
    return y

def gamma (x,v):   #ガンマ変換
    c = 1
    x01 = x/255
    y01 = c * (x01**v)
    y = y01*255
    return y

def bgrcolor (x,b,g,r):  #BGR色調の調整
    f = np.zeros(x.shape)
    f[:,:,0] = gamma(x[:,:,0],b/10)
    f[:,:,1] = gamma(x[:,:,1],g/10)
    f[:,:,2] = gamma(x[:,:,2],r/10)
    return f
    

def filterr(x,n):    #平滑化フィルタ
    con = np.zeros(x.shape)
    if(n == 1):
        con = convolve(x, w3)
    elif(n == 2):
        con = convolve(x, w5)
    else:
        con = x
    return con

def laplacian(x,n):     #ラプラシアンフィルタ
    if(n == 0):
        lap = cv2.Laplacian(x, cv2.CV_32F)
    elif(n == 1):
        lap = cv2.Laplacian(x, cv2.CV_32F,ksize=3)
    elif(n == 2):
        lap = cv2.Laplacian(x, cv2.CV_32F,ksize=5)
    else:
        lap = x
    return lap

def senei(x):   #先鋭化
    ims = convolve(x, se)
    return ims


cv2.namedWindow('image') # create win with win name
cv2.namedWindow('image2')

switch = '0 : gammaOFF \n1 : gammaON'   #ガンマ変換用のスイッチ
cv2.createTrackbar(switch,'image',0,1,myfunc)
cv2.createTrackbar('gamma', # name of value
                   'image', # win name
                   0, # min
                   10, # max
                   myfunc) # callback func

switch3 = '0 : rbgOFF \n1 : rbgON'      #BGR調整用のスイッチ
cv2.createTrackbar(switch3,'image',0,1,myfunc)  

cv2.createTrackbar('B','image',0,255,myfunc)
cv2.createTrackbar('G','image',0,255,myfunc)
cv2.createTrackbar('R','image',0,255,myfunc)

switch1 = '0 : hsvOFF \n1 : hsvON'      #HSV調整用のスイッチ
cv2.createTrackbar(switch1,'image',0,1,myfunc)

switch2 = '0 : filterOFF \n1 : filterON'   #平滑化フィルタ用のスイッチ
cv2.createTrackbar(switch2,'image',0,1,myfunc)

switch4 = '0 : lapOFF \n1 : lapON'  #ラプラシアンフィルタ用のスイッチ
cv2.createTrackbar(switch4,'image',0,1,myfunc)

cv2.createTrackbar('num','image',0,2,myfunc)    #各フィルタにおける段階の切り替え


switch5 = '0 : senOFF \n1 : senON'  #先鋭化用のスイッチ
cv2.createTrackbar(switch5,'image2',0,1,myfunc)

cap = cv2.VideoCapture(0)   #
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)


while(True):

    ret, frame = cap.read()
    if not ret: continue

    gm = cv2.getTrackbarPos('gamma',  # get the value
                           'image')  # of the win
    s = cv2.getTrackbarPos(switch,'image')
    s1 = cv2.getTrackbarPos(switch1,'image')
    s2 = cv2.getTrackbarPos(switch2,'image')
    s3 = cv2.getTrackbarPos(switch3,'image')
    s4 = cv2.getTrackbarPos(switch4,'image')
    s5 = cv2.getTrackbarPos(switch5,'image2')
    b = cv2.getTrackbarPos('B','image')
    g = cv2.getTrackbarPos('G','image')
    r = cv2.getTrackbarPos('R','image')
    n = cv2.getTrackbarPos('num','image')

    ## do something by using v
    
    img_gs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_bgr = cv2.split(frame)
    frame_s = s_tone(frame)
    
    if(s == 1 and s1 == 0 and s2 == 0 and s3 == 0 and s4 == 0 and s5 == 0):
        frame_g = gamma(frame,gm)
        cv2.imshow('image', frame_g)  # show in the win
    elif(s == 0 and s1 == 1 and s2 == 0 and s3 == 0 and s4 == 0 and s5 == 0):
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow('image', frame_hsv)
    elif(s == 0 and s1 == 0 and s2 == 1 and s3 == 0 and s4 == 0 and s5 == 0):
        cv2.imshow('image', filterr(img_gs,n))
    elif(s == 0 and s1 == 0 and s2 == 0 and s3 == 1 and s4 == 0 and s5 == 0):
        cv2.imshow('image', bgrcolor(frame,b,g,r))
    elif(s == 0 and s1 == 0 and s2 == 0 and s3 == 0 and s4 == 1 and s5 == 0):
        cv2.imshow('image', laplacian(img_gs,n))
    elif(s == 0 and s1 == 0 and s2 == 0 and s3 == 0 and s4 == 0 and s5 == 1):
        cv2.imshow('image2', senei(img_gs))
    else:
        cv2.imshow('image', frame)
    
    k = cv2.waitKey(1)
    if k == ord('q') or k == 27:
        break

cap.release()
cv2.destroyAllWindows()