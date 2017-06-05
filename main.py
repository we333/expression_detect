import cv2

def smile_detect():
	face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

	cap = cv2.VideoCapture(0)
	while (True):
		ret,frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		i = 1
		for (x,y,w,h) in faces:
			rect = (x,y,w,h)	
			frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
			face = frame[x:x+w, y:y+h]
			if i == 1:
				print face
				i += 1
			cv2.imshow("camera", frame)
			cv2.imshow('face',face)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

if __name__ == "__main__":
  smile_detect()
