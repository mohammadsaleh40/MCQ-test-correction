import cv2 as cv
import numpy as np

def rescaleFrame (frame , scale = 0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation = cv.INTER_AREA)

def sotone_detect(cnt , s):
    if cv.contourArea(cnt)/cv.arcLength(cnt,True) > int(s*5) and cv.contourArea(cnt)/cv.arcLength(cnt,True) <int(s*12):
        return True
    else:
        return False
        

def changeres (width, height):
    capture.set(3 , width)
    capture.set(4 , height)

def fa(n):
    n = int(n)
    return ((n+1)%2)+n


def miyangin_gir(contours):
    l = len(contours)
    sum_y = 0
    sum_x = 0
    for x in contours:
        sum_x += x[0][0][0]
        sum_y += x[0][0][1]
    return (int(sum_x/l) , int(sum_y/l))

def fasele (a , b):
    
    return (abs(a[0]-b[0])+abs(a[1]-b[1]))

def door_tarin(a , markaz):
    door = a[0][0]
    for i in range (1,len(a)):
        
        if fasele(a[i][0] , markaz) > fasele(door , markaz):
            door = a[i][0]
    return door


def nazdik_tarin (a , markaz):
    nazdik = a[0][0]
    for i in range (1,len(a)):
        
        if fasele(a[i][0][0] , markaz) < fasele(nazdik , markaz):
            nazdik = a[i]
    return nazdik

def nazdik_tarin_m (contours , markaz):
    nazdik = contours[0]
    #nazdik_i = 0
    for i in range (1,len(contours)):
        if fasele(contours[i][0][0], markaz) < fasele(nazdik[0][0], markaz):
            nazdik = contours[i]
            #nazdik_i = i
            
    return nazdik

def predict_to_array(x):
    l = len(x[0])
    print(l)
    for i in range (len(x)):
        a = np.zeros(l)
        
        max_i = 0
        max = x[i][0]
        j = 0
        while j < l:
            if max<x[i][j]:
                max = x[i][j]
                max_i = j
            j+=1
        a[max_i] = 1
        
        x[i] = a
    return x


def predict_to_index(x):
    l = len(x[0])
    khorooji = np.zeros(len(x))
    
    for i in range (len(x)):
        a = np.zeros(l)
        
        max_i = 0
        max = x[i][0]
        j = 0
        while j < l:
            if max<x[i][j]:
                max = x[i][j]
                max_i = j
            j+=1
        a[max_i] = 1
        khorooji [i] = max_i
        x[i] = a
    return khorooji



def chek_shabih(x , y):
    if len(x)!=len(y):
        return False
    for i in range(len(x)):
        
        if x[i]!=y[i]:
            return False
    return True



def tolid_dade_df(contours , list_gooshe_ha , markaz , df):
    dict = {}

    for i in range(115):
        stx = "x"+str(i)
        sty = "y"+str(i)
        dict[stx] = 0
        dict[sty] = 0
    for i in range(len(contours)):
        dict["x"+str(i)] , dict["y"+str(i)] = door_tarin(contours[i] , markaz)-markaz
    if len(list_gooshe_ha) == 4:
        dict["ULx"] , dict["ULy"]= door_tarin(list_gooshe_ha[0] , markaz) - markaz

        dict["URx"] , dict["URy"]= door_tarin(list_gooshe_ha[1] , markaz) - markaz

        dict["DRx"] , dict["DRy"]= door_tarin(list_gooshe_ha[2] , markaz) - markaz

        dict["DLx"] , dict["DLy"]= door_tarin(list_gooshe_ha[3] , markaz) - markaz
    else:
        dict["ULx"] , dict["ULy"] = None , None
        dict["URx"] , dict["URy"] = None , None
        dict["DRx"] , dict["DRy"] = None , None
        dict["DLx"] , dict["DLy"] = None , None


    df = df.append(dict, ignore_index=True)
    return df

def contours2X(contours , markaz):
    
    X = np.zeros((115,2))
    
    for i in range(len(contours)):
        #print(X[i].shape)
        X[i] = door_tarin(contours[i] , markaz)-markaz

    return X

def gooshe_X(X , model):

    x = X.copy()
    for i in range(len(X)):
        x[i] = X[i]/X[i].std()

    y = model.predict(x)

    predict_UL = predict_to_index(y[0])
    predict_UR = predict_to_index(y[1])
    predict_DR = predict_to_index(y[2])
    predict_DL = predict_to_index(y[3])

    khorooji = np.zeros((len(X),4,2))
    for i in range(len(X)):
        
        khorooji[i][0] = X[i][int(predict_UL[i])]
        khorooji[i][1] = X[i][int(predict_UR[i])]
        khorooji[i][3] = X[i][int(predict_DR[i])]
        khorooji[i][2] = X[i][int(predict_DL[i])]
    return khorooji

def warp_image(frame, x ):
    width , hight = [1750, 2000]
    pts2 = np.float32([[0,0],[width , 0],[0 , hight],[width , hight]])
    matrix = cv.getPerspectiveTransform(x,pts2)
    imgoutput = cv.warpPerspective(frame,matrix,(width,hight))
    return imgoutput




def siyah_peyda_kon(img):
    # تشخیص مقیاس عکس
    s = img.shape[1]/ 1080
    
    #تار کردن تصاویر برای رفع نویز
    #بلور کم برای این که فقط نویز بره
    blur_kam = cv.GaussianBlur(img , (fa(s*4),1) , fa(s*2))
    #بلور زیاد برای این که خود پاسخ نامه محو بشه و پس زمینه اگر اجسام مزاحم داره بمونه
    blur_ziyad = cv.GaussianBlur(img , (fa(s*26),fa(s*38)) , int(s*133))


    # مشخص کردن بازه رنگ سیاه
    # عیبی که این بخش داره اینه که رنگ‌های تیره رو هم احتمال داره سیاه تشخیص بده.
    uper_black = np.array([0,0,0])
    lower_black = np.array([100,100,100])

    # تشخیص رنگ‌های سیاه از روی عکس
    mask_kam = cv.inRange(blur_kam, uper_black , lower_black)

    # تشخیص اجسام تیره اطراف
    mask = cv.inRange(blur_ziyad , uper_black , lower_black)

    #بزرگ تر کردن ناحیه اجسام مزاحم اطراف
    kernel = np.ones((fa(s*40),fa(s*60)) , np.uint8)
    mask = cv.dilate (mask , kernel)

    #mask_show = rescaleFrame(mask , s/0.55)
    #cv.imshow('mask1', mask_show)

    # برداشتن ناحیه اجسام تیره از روی سیاهی‌های واقعی تر 
    mask = cv.bitwise_not(mask)
    mask = cv.bitwise_and(mask_kam , mask)

    # یک کم بزرگ تر کردن اجسام سیاه
    ## در این مرحله ستون‌های بارکد مانند هر ۱۰ تای نزدیک به همش باید به هم بچسبند
    kernel = np.ones((fa(s*14),1) , np.uint8)
    mask = cv.dilate(mask , kernel)

    # چند برابر بیشتر از بزرگ کردن قبلی کوچیک کردن اجسام سیاه
    ## تو این مرحله چون ستون‌هایی از اون نقطه‌های سیاه تشکیل دادیم می‌تونیم راحت کوچیک کنیم تا باز هم نقاط سیاه دیگه حذف بشه.
    kernel = np.ones((fa(s*57),fa(s*2)) , np.uint8)
    mask = cv.erode(mask , kernel)


    #mask_show = rescaleFrame(mask , 0.55/s)
    #cv.imshow('mask2', mask_show)

    # بزرگ کردن اجسام سیاه
    ## تو این مرحله با بزرگ کردن این اشیاء باید چند ستون دو طرف برگه درست بشه که از اون نقاط سیاه 10 یا 5 تایی تشکیل شده..
    kernel = np.ones((fa(s*58),fa(s*3)) , np.uint8)
    mask = cv.dilate(mask , kernel)

    #mask_show_1 = rescaleFrame(mask , 0.55/s)

    # اینجا یک بار دیگه با سیاهی‌های صفحه که قبلا تشخیص داده شده بود اشتراک گرفته می‌شه.
    mask = cv.bitwise_and(mask_kam , mask)



    # اینجا این قدر بزرگ می‌شه که دو ستون دو طرف برگه درست بشه.
    kernel = np.ones((fa(s*62),fa(s*2)) , np.uint8)
    mask = cv.dilate(mask , kernel)

    #mask_show_2 = rescaleFrame(mask , 0.55/s)




    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # یک خبر خوب اینه که اینجا ما عدد دو رو داشته باشیم.
    if 2 < len(contours):
        l = 0
        for i in range (0,len(contours)):
            if sotone_detect (contours[i] , s ):
                l+=1
                if 'sotoon_R' not in locals() and 'sotoon_L' not in locals():
                    sotoon_R = contours[i]

                elif 'sotoon_R' in locals() and 'sotoon_L' not in locals():
                    if cv.arcLength(contours[i] , True) > cv.arcLength(sotoon_R , True):

                        sotoon_L = sotoon_R.copy()
                        sotoon_R = contours[i]
                    else:
                        sotoon_L = contours[i]
                        
                else:

                    if cv.arcLength(contours[i] , True) > cv.arcLength(sotoon_R , True):
                        sotoon_L = sotoon_R
                        sotoon_R = contours[i]
                    elif cv.arcLength(contours[i] , True) > cv.arcLength(sotoon_L , True):
                        sotoon_L = contours[i]
        
        # اگر نشد حداقل اینجا ببینیم 2 اومده.
        
        if l != 2:
            #print text on img
            cv.putText(img , str(l)+' sotoon' , (img.shape[1]//4 , img.shape[0]//2) , cv.FONT_HERSHEY_SIMPLEX , s*2.2 , (0,0,255) , 3)
            
            return img , mask, False
    elif 2 == len(contours):
        sotoon_L = contours[0]
        sotoon_R = contours[1]    
    else:
        #print text on img
        cv.putText(img , 'No sotoon' , (img.shape[1]//5, img.shape[0]//5) , cv.FONT_HERSHEY_SIMPLEX , fa(s*2) , (0,0,255) , fa(s*2))
        return img, mask, False

    like_img = np.zeros(img.shape[:2] , np.uint8)

    cv.drawContours(like_img, [sotoon_R], 0 ,255, -1)
    cv.drawContours(like_img, [sotoon_L], 0 , 255 , -1)

    like_img = cv.bitwise_and(like_img , mask_kam)
    contours, hierarchy = cv.findContours(like_img, cv.RETR_EXTERNAL , cv.CHAIN_APPROX_SIMPLE)
    
    markaz = miyangin_gir(contours)

    like_img = img.copy()
    
    for x in contours:
        a = door_tarin(x , markaz)
        cv.circle(like_img , a , fa(s*4) , (0,0,255) , -1)
    #کشیدن دایره روی مرکز
    cv.circle(like_img, markaz, fa(s*7) , (100,200,255) , -1)
    
    # اگر تعداد تشخیصی مشکل حاد داشت
    if len(contours) < 110 or len(contours) > 115:
        #print text on img
        cv.putText(like_img , 'خرابه' , (img.shape[1]//5, img.shape[0]//5) , cv.FONT_HERSHEY_SIMPLEX , fa(s*2) , (0,0,255) , fa(s*2))
        return like_img, mask, False

    # خلاصه اگر همه چی خراب بود این ماجرا
    return like_img , mask, contours



def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
