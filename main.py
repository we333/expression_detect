# -- coding: utf-8 --

import cv2
import evaluate

def smile_detect():
	i = 0
	pred = 0
	face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

	cap = cv2.VideoCapture(0)
	while (True):
		ret,frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		i += 1
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)

		for (x,y,w,h) in faces:
			rect = (x,y,w,h)
			frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
			face = frame[x:x+w, y:y+h]

			if i %10 == 0:
				pred = evaluate.evaluate_image(face)
			
			cv2.putText(frame,'smile:%f'%pred,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)

			cv2.imshow("camera", frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

if __name__ == "__main__":
  smile_detect()
