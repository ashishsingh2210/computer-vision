import cv2
import math
import mediapipe as mp
#ffpyplayer for playing audio
# from ffpyplayer.player import MediaPlayer



# read webcam to detect hand
cap = cv2.VideoCapture(0)

# read video from folder
video = cv2.VideoCapture('sample.mp4')

# play audio of the video
# player = MediaPlayer('sample.mp4')

# detect hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.6, 
                      min_tracking_confidence=0.8)

# for drawing handlandmarks
mpDraw = mp.solutions.drawing_utils

############## initialize varibles ###########
xt,yt, xi,yi, xm,ym = 0,0, 0,0, 0,0  # tip of thumb, index, middle fingure 
x1,y1,x2,y2 = 0,0, 0,0     # corrdinates for drawing rectangle
lengthTI = 0             # length of thumb and index fingure
lengthIM = 0             # length of index and midle fingure
h,w,c = 0,0,0            # image/ video height, width, channel(RGB)
i=0
while True:
    # read webcam for hand recongnition
    success,img = cap.read()
    
    # read video
    successV,vid = video.read()
    
    # read audio file in video
    # audio_frame, audio = player.get_frame()
    
    # fliping webcam
    img = cv2.flip(img,1)
    
    # resize img and video in same size
    # while taking corrdinates for screenshot
    # we will detect hand from webcam which is used for taking corrdinates then 
    ## those corrdinates will be used in video to take screenshot
    img =cv2.resize(img,(1300,720),fx=0,fy=0)
    vid =cv2.resize(vid,(1300,720),fx=0,fy=0)
    
    # convert BGR to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # process webcam for recogniting 
    results = hands.process(imgRGB)
    
    # detect landmarks
    if results.multi_hand_landmarks:
        # print(results.multi_hand_landmarks)
        for handlms in results.multi_hand_landmarks:
            # print(handlms)
            
            # detect landsmarks along with there id
            for id, lm in enumerate(handlms.landmark):
                # print(id,lm)
                h, w, c = img.shape
                
                # denormalizing the normalized landmarks 
                #(landmarks we get from hand are on in form of matrix (width and height of image))
                cx,cy = int(lm.x*w), int(lm.y*h)
                
                # get thumb tip and corrdinates
                if id == 4:
                    xt,yt = cx,cy
                    
                    # draw circle/ point to view location
                    # we draw on img not on video and it is not compulsary
                    cv2.circle(img, (cx, cy), 3, (0, 255, 255), cv2.FILLED)

                # get thumb tip and corrdinates
                if id == 8:
                    xi,yi = cx,cy

                    # draw circle/ point to view location
                    # we draw on video not on img
                    cv2.circle(vid, (cx, cy), 3, (0, 255, 255), cv2.FILLED)

                # get thumb tip and corrdinates
                if id == 12:
                    xm,ym = cx,cy

                    # draw circle/ point to view location
                    # we draw on img not on video and it is not compulsary
                    cv2.circle(img, (cx,cy), 3, (0, 255, 255), cv2.FILLED)
    
    # find length between thumb and index and this length will be used
    ## for taking screenshot
    lengthTI = math.hypot(xi-xt, yi-yt)

    # find length between index and middle and this length will be used
    ## for selecting the area for taking screenshot
    lengthIM = math.hypot(xi-xm,yi-ym)

    # here we check wheather index and middle fingure are together or not
    # it they are together and then we take 1st corrdinate of the retangle and start drawing rectangle
    # if index and middle fingure are not close the we stop drawing rectangle and point where we stop
    ### , that will be our 2nd corrdinate
    if lengthIM < 30:
        
        # this condition is to used to take 1st corrdinate and also fix the corrdinate
        ## if we don't do that then corrdinate will vary and we wouldn't be able to make the static/fix rectangle
        if  x1==0 or y1==0:
            x1,y1 = xi,yi
            
        # this condition will help us to vary size of retangle with fix 1st corrdinate
        if x1!=0 and y1!=0:
            x2,y2 = xi,yi        

   # draw rectangle
    cv2.rectangle(vid,(x1,y1),(x2,y2),255)
    
    # taking screenshot
    # we check if rectangle corrdinate are not zero
    ## if not zero then we crop the image in range of corrdinates
    ### then if image in not empty
    #### then we save the screenshot in our local computer
    ### if img is empty then we print the error with image size
    ##### once screenshot is captured we reset the corrdinate to zero so that we get next desired corrdinate
    if (x1 != 0 and y1 != 0) or (x2 != 0 and y2 != 0):
        img1 = vid[ y1:y2 , x1:x2]
        if 10 < lengthTI <30:
            try:
                cv2.imwrite(f'/home/ubuntu/python learning/codes/python-spyder coding/ss/ss{str(i)}.jpg',img1)
                i=i+1
                cv2.putTExt(img,'screenShot captured',(0,101),
                            cv2.FONT_HERSHEY_DUPLEX,
                            1,(255,0,0),2)
                x1,y1,x2,y2 = 0,0,0,0
            except:
                print(f'TRY AGAIN - screen shot is not take \n because image size is {img1.shape}')
                x1,y1,x2,y2 = 0,0,0,0
    
    # show webcam
    # cv2.imshow('image',img)

    # show video
    cv2.imshow('video',vid)      
    
    # this for for audio frame
    # if audio != 'eof' and audio_frame is not None:
    # #audio
        # aud, t = audio_frame
    
    k = cv2.waitKey(1)
    
    # space key to pause and play
    ##### NOTE: we can pause using space key but we can play again by pressing any key ####
    ##### NOTE: pause in opnecv means stop taking frames which means 
    #########   not opreation will pe played when we have pause the video
    if k == 32:
        cv2.waitKey(0)
        
        # pause the audio
        # player.set_pause(False)
 
    # play the video and audio
    # if k == ord('p'):
    #     cv2.waitKey(1)
    #     # player.set_pause(True)
    if k == ord('q'):
        break
    if not successV:
        print('end of video')
        break
    


# player.close_player()
cap.release()
cv2.destroyAllWindows()
