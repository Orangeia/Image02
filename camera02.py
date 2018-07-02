import numpy as np
import cv2
import math
from scipy.ndimage.filters import convolve
N = 3
w = np.ones((N,N)) / float(N*N)

def myfunc(i):
    pass # do nothing

def s_tone (x):
    y = (np.sin(np.pi * (x/255 - 0.5)) + 1)/2 * 255
    return y

def gamma (x,v):
    #print(v)
    c = 1
    x01 = x/255
    y01 = c * (x01**v)
    y = y01*255
    return y

def rgbcolor (x,r,g,b,s):
    if(s == 1):
        y = x[:,:,0]
        return y
    else:
        return x
    
def filterr(im,w):
    imf = convolve(im, w)
    return imf

cv2.namedWindow('image') # create win with win name



switch = '0 : gammaOFF \n1 : gammaON'
cv2.createTrackbar(switch,'image',0,1,myfunc)
cv2.createTrackbar('gamma', # name of value
                   'image', # win name
                   0, # min
                   10, # max
                   myfunc) # callback func



switch1 = '0 : hsvOFF \n1 : hsvON'
cv2.createTrackbar(switch1,'image',0,1,myfunc)

switch2 = '0 : filterOFF \n1 : filterON'
cv2.createTrackbar(switch2,'image',0,1,myfunc)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)


while(True):

    ret, frame = cap.read()
    if not ret: continue
    #print(frame)


    gm = cv2.getTrackbarPos('gamma',  # get the value
                           'image')  # of the win
    s = cv2.getTrackbarPos(switch,'image')
    s1 = cv2.getTrackbarPos(switch1,'image')
    s2 = cv2.getTrackbarPos(switch2,'image')

    ## do something by using v
    
    img_gs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_bgr = cv2.split(frame)
    frame_s = s_tone(frame)
    frame_rgb = rgbcolor(frame,r,g,b,s)
    
    if(s == 1 and s1 == 0 and s2 == 0):
        frame_g = gamma(frame,gm)
        cv2.imshow('image', frame_g)  # show in the win
    elif(s == 0 and s1 == 1 and s2 == 0):
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow('image', frame_hsv)
    elif(s == 0 and s1 == 0 and s2 == 1):
        cv2.imshow('image', filterr(img_gs,w))
    else:
        cv2.imshow('image', frame)
    
    k = cv2.waitKey(1)
    if k == ord('q') or k == 27:
        break



cap.release()
cv2.destroyAllWindows()