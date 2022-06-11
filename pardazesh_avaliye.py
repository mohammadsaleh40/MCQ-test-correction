from time import sleep
import cv2 as cv
import numpy as np
import pandas as pd
#from math import abs

from functions import fa , miyangin_gir , door_tarin , fasele , nazdik_tarin , nazdik_tarin_m
from functions import siyah_peyda_kon , stackImages , sotone_detect , changeres , rescaleFrame 
from functions import tolid_dade_df

img = cv.imread('IMG_4.jpg')

img2 , mask , contours = siyah_peyda_kon(img)

cv.imshow('img' , rescaleFrame(img, 0.2))

cv.imshow('img2' , rescaleFrame(img2, 0.2))
cv.imshow('mask' , rescaleFrame(mask, 0.2))


if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()

