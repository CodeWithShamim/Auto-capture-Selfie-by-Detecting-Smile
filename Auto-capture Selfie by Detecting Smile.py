import cv2
import datetime

video = cv2.VideoCapture(0)
haar_cascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')




while True:
    shamim, img = video.read()
    original_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detect_face = haar_cascade.detectMultiScale(gray, 1.3, 4)
    

    
    for (x, y, w, h) in detect_face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (20, 255, 79), 3)



        img_roi = img[y:y+h, x:x+w]
        gray_roi = gray[y:y+h, x:x+w]
        detect_smile = smile_cascade.detectMultiScale(gray_roi, 1.3, 5)
        for (x1, y1, w1, h1) in detect_smile:
           cv2.rectangle(img_roi, (x1, y1), (x1+w1, y1+h1), (255, 255, 250), 2)
           time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
           #--------------
           file = f'smile_pic-{time_stamp}.jpg'
           cv2.imwrite(file, original_img)
           
    cv2.imshow("img", img)        
    if cv2.waitKey(10) &0xFF == ord('q'):
      break      

     
   
  
