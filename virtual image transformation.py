import cv2
import numpy as np
# import math
# import time
import mediapipe as mp

#################################################
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
h,w,c = 480,640,3

############## initialize variables ########################
I = np.float32([[1,0,0],[0,1,0],[0,0,1]])   # identity matrix
l,r,u,d=0,0,0,0   # translation left, right, up, down
in_,out = 0,0     # zoom in and out
wd,ht = 0,0       # crop width and height
angle = np.radians(0)
MZI,MZO = I,I
MTU,MTD,MTR,MTL = I,I,I,I
MR = I
MRF = I
M = I

############  hand recognitation  ####################
max_hand = 1
min_detect_confi = 0.8
min_track_confi = 0.9
mpHand = mp.solutions.hands
hands = mpHand.Hands(max_num_hands=max_hand,
                     min_detection_confidence=min_detect_confi,
                     min_tracking_confidence=min_track_confi)
mpdraw = mp.solutions.drawing_utils

########### index fingure coordinates ###############
x,y = 0,0

############## image transfomations #################
l = 100  # length of rectangle

## scaling/zoom in and out
xz1,yz1,xz2,yz2 = 0,0,100,50  # coordinates for zoom/scaling box
colorZ = (160,0,0)
###### zooom in
xzi1,yzi1,xzi2,yzi2 = xz1,yz2//2,xz2//2-7,yz2//2+yz2
colorZI = (160,0,0)
###### zooom out
xzo1,yzo1,xzo2,yzo2 = xzi2,yzi1,xzi2 + xz2//2-7,yzi2
colorZO = (160,0,0)

## rotation
xr1,yr1,xr2,yr2 = xz2,0,xz2+l,yz2 # cordinates for rotation box
colorR = (160,0,0)

## cropping
xc1,yc1,xc2,yc2 = xr2,0,xr2+l,yr2 # cordinates for cropping box
colorC = (160,0,0)
###### crop Width
xcw1,ycw1,xcw2,ycw2 = xc1,yc2//2,xc2-(xc2-xc1)//2,yc2//2+yc2
colorZI = (160,0,0)
###### crop Height
xch1,ych1,xch2,ych2 = xcw2,ycw1,xc2-7,ycw2
colorZO = (160,0,0)

## reflection
xrf1,yrf1,xrf2,yrf2 = xc2,0,xc2+l,yc2 # cordinates for reflection box
colorRF = (160,0,0)

## translation
xt1,yt1,xt2,yt2 = xrf2,0,xrf2+l,yrf2 # cordinates for translation box
colorT = (160,0,0)
#### buttons
##### "UP"
xtu1,ytu1,xtu2,ytu2 = xt2+40,0,xt2+80,yt2+10
colorTUp = (160,0,0)

##### "down"
xtd1,ytd1,xtd2,ytd2 = xtu1,ytu2,xtu2,ytu2+50
colorTDown = (160,0,0)

##### "left"
xtl1,ytl1,xtl2,ytl2 = xtu1-40,ytd1-20,xtd1,ytd2-25
colorTRight = (160,0,0)

##### "right"
xtr1,ytr1,xtr2,ytr2 = xtu2,ytd1-20,xtd2+40,ytd2-25
colorTLeft = (160,0,0)

#################################################
img2 = cv2.imread('/home/ubuntu/python learning/codes/python-spyder coding/images/P_20200513_001437_EFF.jpg')
#################################################


def translation():
    
############ translation up arrow
            cv2.rectangle(mask, (xtu1,ytu1), (xtu2,ytu2), colorTUp,-1)
            cv2.line(mask,(xtu1,ytu2),(xtu2,ytu2),(0,180,250),7)
            cv2.putText(mask,'U',(xtu1+11,ytu2-15),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
            
############ translation down arrow
            cv2.rectangle(mask, (xtd1,ytd1), (xtd2,ytd2), colorTDown,-1)
            cv2.putText(mask,'D',(xtd1+11,ytd2-15),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
            
############ translation left arrow
            cv2.line(mask,(xtl2,ytl1),(xtl2,ytl2),(0,180,250),7)
            cv2.rectangle(mask, (xtl1,ytl1), (xtl2,ytl2), colorTLeft,-1)
            cv2.putText(mask,'L',(xtl1+10,ytl2-15),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
            
############ translation right arrow
            cv2.line(mask,(xtr1,ytr1),(xtr1,ytr2),(0,180,250),7)
            cv2.rectangle(mask, (xtr1,ytr1), (xtr2,ytr2), colorTRight,-1)
            cv2.putText(mask,'R',(xtr1+10,ytr2-15),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)



while True:
    
    frame,img = cap.read()
    img = cv2.flip(img,1)
#########################################################################################

    blackbg = np.zeros(img.shape[:2],dtype="uint8")
    mask = cv2.bitwise_and(img, img, mask = blackbg)
    
#########################################################################################
    
#########################################################################################

    img2 = cv2.resize(img2,((int(mask[50:mask.shape[0],0:mask.shape[1]].shape[0]//1.5),
                              int(mask[0:mask.shape[0],0:mask.shape[1]].shape[1]//1.5)))) 
    whiteBG = np.zeros((mask[50:mask.shape[0],
                  0:mask.shape[1]].shape[0],
             mask[0:mask.shape[0],
                  0:mask.shape[1]].shape[1],
             3),dtype="uint8")
    
    whiteBG[whiteBG.shape[0]-img2.shape[0]:whiteBG.shape[0]+img2.shape[0],
            whiteBG.shape[1]//2-img2.shape[1]//2:whiteBG.shape[1]//2+img2.shape[1]//2] = img2
   
    mask[50:whiteBG.shape[0]+50,0:mask.shape[1]] = whiteBG
    M = np.linalg.multi_dot([MZI,MZO,MTU,MTD,MTR,MTL,MRF,MR])
    mask = cv2.warpPerspective(whiteBG, M, 
                                (whiteBG.shape[1],whiteBG.shape[0]))

#########################################################################################
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)



    result = hands.process(imgRGB)
    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:

            for id,lm in enumerate(handlms.landmark):
                # h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
 
                if id == 8:
                    x,y = cx,cy
                    cv2.circle(mask,(x,y),5,(0,255,255),-1)
                    
#################### zoom
                    if xz1<x<xz2 and yz1<y<yzo2:
                            colorZ = (160,100,0)
                            
############################ zoom in     
                            cv2.rectangle(mask, (xzi1,yzi1), (xzi2,yzi2), colorZI,-1)
                            cv2.line(mask,(xzi2,yzi1),(xzi2,yzi2),(0,180,250),9)
############################ zoom out
                            cv2.rectangle(mask, (xzo1,yzo1), (xzo2,yzo2), colorZO,-1)                            
############################ zoom in and out text
                            cv2.putText(mask,'-',(xzo1+10,yz2+yzi2//4),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),3)
                            cv2.putText(mask,'+',(xzi1+10,yz2+yzi2//4),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),3)
                         
                            
############################## zoom in                         
                            if xzi1<x<xzi2 and yzi1<y<yzi2:
                                    colorZI = (160,100,0)
                                    MZI = np.float32([[1.5+in_, 0 , 0],
                                                    [0, 1.5+in_ , 0],
                                                    [0,  0  , 1]])
                                    in_ = in_+0.01
                                    # M = np.linalg.multi_dot([MZI,MZO,MTU,MTD,MTR,MTL,MRF,MR])
                                    # mask = cv2.warpPerspective(whiteBG, M, 
                                    #                             (whiteBG.shape[1],whiteBG.shape[0]))
                                    cv2.line(mask,(xzi2,yzi1),(xzi2,yzi2),(0,180,250),9)
                                    cv2.rectangle(mask, (xzi1,yzi1), (xzi2,yzi2), colorZI,-1)
                                    cv2.putText(mask,'+',(xzi1+10,yz2+yzi2//4),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),3)
                            else:
                                    colorZI = (160,00,0)
                                    
############################## zoom out
                            if xzo1<x<xzo2 and yzo1<y<yzo2:
                                    colorZO = (160,100,0)
                                    MZO = np.float32([[1+out, 0 , 0],
                                                    [0, 1+out , 0],
                                                    [0,  0  , 1]])
                                    out = out-0.007
                                    # M = np.linalg.multi_dot([MZI,MZO,MTU,MTD,MTR,MTL,MRF,MR])
                                    # mask = cv2.warpPerspective(whiteBG, M, 
                                    #                             (whiteBG.shape[1],whiteBG.shape[0]))
                                    cv2.line(mask,(xzi2,yzi1),(xzi2,yzi2),(0,180,250),9)
                                    cv2.rectangle(mask, (xzo1,yzo1), (xzo2,yzo2), colorZO,-1)
                                    cv2.putText(mask,'-',(xzo1+10,yz2+yzi2//4),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),3)
        
                            else:
                                    colorZO = (160,00,0)
                                           
                          
                    else:
                            colorZ = (160,00,0)   
                                      
                    
##################### roation 
                    if xr1<x<xr2 and yr1<y<yr2:
                            colorR = (160,100,0)
                            
                            in_ = in_+0.009
                            MR = np.float32([[np.cos(angle), -(np.sin(angle)), 0],
                                            [np.sin(angle), np.cos(angle), 0],
                                            [0, 0, 1]])
                            angle = angle+np.radians(0.5)
                            # M = np.linalg.multi_dot([MZI,MZO,MTU,MTD,MTR,MTL,MRF,MR])
                            # mask = cv2.warpPerspective(img2, M, 
                            #                             (whiteBG.shape[1],whiteBG.shape[0]))
                    else:
                            colorR = (160,00,0)
                    
                    
##################### croping
                    if xc1<x<xc2 and yc1<y<ych2:
                            colorC = (160,100,0)
############################ crop width     
                            cv2.rectangle(mask, (xcw1,ycw1), (xcw2,ycw2), colorZI,-1)
############################ crop height
                            cv2.rectangle(mask, (xch1,ych1), (xch2,ych2), colorZO,-1)                            
                            cv2.line(mask,(xcw2,ycw1),(xcw2,ycw2),(0,180,250),9)
############################ crop W and H text
                            cv2.putText(mask,'W',(xcw1+10,yc2+ycw2//4),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),3)
                            cv2.putText(mask,'H',(xch1+10,yc2+ych2//4),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),3)
                         
                            
############################## crop width                     
                            if xcw1<x<xcw2 and ycw1<y<ycw2:
                                    colorZI = (160,100,0)
                                    cv2.rectangle(mask,(xz1+wd,yz2+ht),(whiteBG.shape[0]-wd,img2.shape[0]-ht),(0,0,255),5)
                                    wd = wd+1
                                    cv2.rectangle(mask, (xcw1,ycw1), (xcw2,ycw2), colorZI,-1)
                                    cv2.putText(mask,'W',(xcw1+10,yc2+ycw2//4),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),3)
                            else:
                                    colorZI = (160,00,0)
                                    
############################## crop height
                            if xch1<x<xch2 and ych1<y<ych2:
                                    colorZO = (160,100,0)
                                    cv2.rectangle(mask,(xz1+wd,yz2+ht),(whiteBG.shape[0]-wd,img2.shape[0]-ht),(0,0,255),5)
                                    ht = ht+1
                                    cv2.rectangle(mask, (xch1,ych1), (xch2,ych2), colorZO,-1)
                                    cv2.putText(mask,'H',(xch1+10,yc2+ych2//4),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),3)
        
                            else:
                                    colorZO = (160,00,0)
                                           
                          
                    else:
                            colorC = (160,00,0)
                    
                    
##################### reflection
                    if xrf1<x<xrf2 and yr1<y<yrf2:
                            colorRF = (160,100,0)
                            MRF = np.float32([[1, 0, 0],
                                            [0, -1, img2.shape[1]],
                                            [0, 0, 1]])
                            # M = np.linalg.multi_dot([MZI,MZO,MTU,MTD,MTR,MTL,MRF,MR])
                            # mask = cv2.warpPerspective(img2, M, 
                            #                             (whiteBG.shape[1],whiteBG.shape[0]))
                    else:
                            colorRF = (160,00,0)

##################### translation
                    if xt1<x<xtr2+10 and yt1<y<ytd2+5:
                            colorT = (160,100,0)
                            translation()
                            if xt1 <x< w and yt1<y<ytd2+10:
###################################### translation up
                                    if xtu1<x<xtu2 and ytu1<y<ytu2:
                                            colorTUp = (160,100,0)
                                            colorT = (160,100,0) 
                                            MTU = np.float32([[1, 0, 0],
                                                            [0, 1, -u],
                                                            [0, 0, 1]])
                                            u = u+1.5
                                            
                                            # M = np.linalg.multi_dot([MZI,MZO,MTU,MTD,MTR,MTL,MRF,MR])
                                            # mask = cv2.warpPerspective(img2, M, 
                                            #                             (whiteBG.shape[1],whiteBG.shape[0]))
                                            cv2.rectangle(mask, (xtu1,ytu1), (xtu2,ytu2), colorTUp,-1)
                                            cv2.line(mask,(xtu1,ytu2),(xtu2,ytu2),(0,180,250),7)
                                            cv2.rectangle(mask, (xtd1,ytd1), (xtd2,ytd2), colorTDown,-1)
                                            cv2.line(mask,(xtl2,ytl1),(xtl2,ytl2),(0,180,250),7)
                                            cv2.line(mask,(xtr1,ytr1),(xtr1,ytr2),(0,180,250),7)
                                            cv2.rectangle(mask, (xtr1,ytr1), (xtr2,ytr2), colorTRight,-1)
                                            cv2.rectangle(mask, (xtl1,ytl1), (xtl2,ytl2), colorTLeft,-1)
                                            cv2.putText(mask,'U',(xtu1+11,ytu2-15),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
                                            cv2.putText(mask,'D',(xtd1+11,ytd2-15),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
                                            cv2.putText(mask,'L',(xtl1+10,ytl2-15),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
                                            cv2.putText(mask,'R',(xtr1+10,ytr2-15),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)

                                    else:
                                            colorTUp = (160,00,0)
                
###################################### translation down
                                    if xtd1<x<xtd2 and ytd1<y<ytd2:
                                            colorTDown = (160,100,0)
                                            colorT = (160,100,0)
                                            MTD = np.float32([[1, 0, 0],
                                                            [0, 1, +d],
                                                            [0, 0, 1]])
                                            d = d+1.5

                                            # M = np.linalg.multi_dot([MZI,MZO,MTU,MTD,MTR,MTL,MRF,MR])
                                            # mask = cv2.warpPerspective(img2, M, 
                                            #                             (whiteBG.shape[1],whiteBG.shape[0]))
                                            cv2.rectangle(mask, (xtu1,ytu1), (xtu2,ytu2), colorTUp,-1)
                                            cv2.line(mask,(xtu1,ytu2),(xtu2,ytu2),(0,180,250),7)
                                            cv2.rectangle(mask, (xtd1,ytd1), (xtd2,ytd2), colorTDown,-1)
                                            cv2.line(mask,(xtl2,ytl1),(xtl2,ytl2),(0,180,250),7)
                                            cv2.line(mask,(xtr1,ytr1),(xtr1,ytr2),(0,180,250),7)
                                            cv2.rectangle(mask, (xtr1,ytr1), (xtr2,ytr2), colorTRight,-1)
                                            cv2.rectangle(mask, (xtl1,ytl1), (xtl2,ytl2), colorTLeft,-1)
                                            cv2.putText(mask,'U',(xtu1+11,ytu2-15),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
                                            cv2.putText(mask,'D',(xtd1+11,ytd2-15),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
                                            cv2.putText(mask,'L',(xtl1+10,ytl2-15),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
                                            cv2.putText(mask,'R',(xtr1+10,ytr2-15),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
                                    else:
                                            colorTDown = (160,00,0)
                
###################################### translation left
                                    if xtl1<x<xtl2 and ytl1<y<ytl2:
                                            colorTLeft = (160,100,0)
                                            colorT = (160,100,0)
                                            MTL = np.float32([[1, 0, -l],
                                                            [0, 1, 0],
                                                            [0, 0, 1]])
                                            l = l+1.5

                                            # M = np.linalg.multi_dot([MZI,MZO,MTU,MTD,MTR,MTL,MRF,MR])
                                            # mask = cv2.warpPerspective(img2, M, 
                                            #                             (whiteBG.shape[1],whiteBG.shape[0]))
                                            cv2.rectangle(mask, (xtd1,ytd1), (xtd2,ytd2), colorTDown,-1)
                                            cv2.line(mask,(xtu1,ytu2),(xtu2,ytu2),(0,180,250),7)
                                            cv2.rectangle(mask, (xtu1,ytu1), (xtu2,ytu2), colorTUp,-1)
                                            cv2.line(mask,(xtl2,ytl1),(xtl2,ytl2),(0,180,250),7)
                                            cv2.line(mask,(xtr1,ytr1),(xtr1,ytr2),(0,180,250),7)
                                            cv2.rectangle(mask, (xtr1,ytr1), (xtr2,ytr2), colorTRight,-1)
                                            cv2.rectangle(mask, (xtl1,ytl1), (xtl2,ytl2), colorTLeft,-1)
                                            cv2.putText(mask,'U',(xtu1+11,ytu2-15),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
                                            cv2.putText(mask,'D',(xtd1+11,ytd2-15),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
                                            cv2.putText(mask,'L',(xtl1+10,ytl2-15),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
                                            cv2.putText(mask,'R',(xtr1+10,ytr2-15),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
                                            
                                            
                                    else:
                                            colorTRight = (160,00,0)
                
###################################### translation right
                                    if xtr1<x<xtr2 and ytr1<y<ytr2:
                                            colorTRight = (160,100,0)
                                            colorT = (160,100,0)
                                            MTR = np.float32([[1, 0,r],
                                                            [0, 1, 0],
                                                            [0, 0, 1]])
                                            r = r+1.5

                                            # M = np.linalg.multi_dot([MZI,MZO,MTU,MTD,MTR,MTL,MRF,MR])
                                            # mask = cv2.warpPerspective(img2, M, 
                                            #                             (whiteBG.shape[1],whiteBG.shape[0]))
                                            cv2.rectangle(mask, (xtd1,ytd1), (xtd2,ytd2), colorTDown,-1)
                                            cv2.line(mask,(xtu1,ytu2),(xtu2,ytu2),(0,180,250),7)
                                            cv2.rectangle(mask, (xtu1,ytu1), (xtu2,ytu2), colorTUp,-1)
                                            cv2.line(mask,(xtl2,ytl1),(xtl2,ytl2),(0,180,250),7)
                                            cv2.line(mask,(xtr1,ytr1),(xtr1,ytr2),(0,180,250),7)
                                            cv2.rectangle(mask, (xtr1,ytr1), (xtr2,ytr2), colorTRight,-1)
                                            cv2.rectangle(mask, (xtl1,ytl1), (xtl2,ytl2), colorTLeft,-1)
                                            cv2.putText(mask,'U',(xtu1+11,ytu2-15),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
                                            cv2.putText(mask,'D',(xtd1+11,ytd2-15),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
                                            cv2.putText(mask,'L',(xtl1+10,ytl2-15),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
                                            cv2.putText(mask,'R',(xtr1+10,ytr2-15),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
                                            

                                    else:
                                            colorTLeft = (160,00,0)
                    else:
                        colorT = (160,00,0)

##########    drawing rectangles for columns    ##########################################
##### zoom
    cv2.rectangle(mask, (xz1,yz1), (xz2,yz2), colorZ,-1)
    cv2.line(mask,(xz2,yz1),(xz2,yz2),(0,180,250),9)
    cv2.putText(mask,'zoom',(xz1+10,yz2//2),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
    
##### rotation
    cv2.rectangle(mask, (xr1,yr1), (xr2,yr2), colorR,-1)
    cv2.line(mask,(xr2,yr1),(xr2,yr2),(0,180,250),9)
    cv2.putText(mask,'rotation',(xr1+10,yr2//2),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
    
##### cropping
    cv2.rectangle(mask, (xc1,yc1), (xc2,yc2), colorC,-1)
    cv2.line(mask,(xc2,yc1),(xc2,yc2),(0,180,250),9)
    cv2.putText(mask,'crop',(xc1+10,yc2//2),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
    
##### reflection
    cv2.rectangle(mask, (xrf1,yrf1), (xrf2,yrf2), colorRF,-1)
    cv2.line(mask,(xrf2,yrf1),(xrf2,yrf2),(0,180,250),9)
    cv2.putText(mask,'reflection',(xrf1+10,yrf2//2),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
    
##### translation
    cv2.rectangle(mask, (xt1,yt1), (xt2,yt2), colorT,-1)
    cv2.putText(mask,'translation',(xt1+10,yt2//2),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
        
################################################
    
    cv2.circle(mask,(x,y),5,(0,255,255),-1)

    
    M = np.linalg.multi_dot([MZI,MZO,MTU,MTD,MTR,MTL,MRF,MR])
    
    cv2.imshow('mask',mask)
        
    # else:
    #     print('in')
    #     cv2.imshow('mask',mask1)
        
    # cv2.imshow('image',img2)
    
    k= cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



