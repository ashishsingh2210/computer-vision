import cv2
import mediapipe as mp
import time
import math
import numpy as np

cap = cv2.VideoCapture(0)
#cap.set(3, 640)
#cap.set(4, 480)

# loading hand module from mediapipe 
mpHands = mp.solutions.hands

# loading hand detection from mediapipe hand module
hands = mpHands.Hands(static_image_mode=False,max_num_hands=1,min_detection_confidence=0.6, min_tracking_confidence=0.8)

# loading the drawring utilits for drawing hand conections
mpDraw = mp.solutions.drawing_utils

x1,y1,x2,y2 = 0,0,0,0  #cordinated for calculating length/ radius of circle
a1,a2 = 0,0 # finding centre point of the circle
length = 0  # radius of circle 
mask = None # mask on image
while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    
    # convert bgr to rgb
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # detect the hand from the image or webcam
    results = hands.process(imgRGB)
    # results.multi_hand_landmarks =  used to get the landmarks or corrdinates of the 
    # hand
    # there are total 21 landmarks (0-20) in that [8,12,16,20] are the tips of fingure and
    # 4 is tip of thumb 
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape                
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # getting corrdinates of tip of thumb
                if id == 4:
                    x1,y1 = cx,cy 
                
                # getting corrdinates of tip of index fingure and making cirle of it
                if id == 8:
                    cv2.circle(img, (cx, cy), 3, (0, 255, 255), cv2.FILLED)
                    x2,y2 = cx,cy
                    if handLms == 8:
                        mpDraw.draw_landmarks(img, handLms)  #, mpHands.HAND_CONNECTIONS
    
    # calculate the length/ radius of the circle
    length = math.hypot(x2 - x1, y2 - y1)
    
    # create a black backgroud
    m = np.zeros(img.shape[:2], dtype="uint8")
    
    # a mask is the same size as our image, but has only two pixel
    # values, 0 and 255 -- pixels with a value of 0 (background) are
    # ignored in the original image while mask pixels with a value of
    # 255 (foreground) are allowed to be kept
    # -1 = use to fill the circle
    cv2.circle(m,(x2,y2),int(length),255,-1)
    cv2.putText(m, str(int(length)), (a1,a2), cv2.FONT_HERSHEY_PLAIN, 2,(0, 255, 100), 2)
    
    # apply our mask -- notice how only the circle in the webcam is cropped out
    mask = cv2.bitwise_and(img, img, mask = m)
    cv2.imshow('mask',mask)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# destroy all windows    
cv2.destroyAllWindows()
# close the webcam
cap.release() 