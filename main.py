import cv2

def smile_detect():
	cap = cv2.VideoCapture(0)
	while (True):
		ret,frame = cap.read()
		print ret
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.imshow("camera", frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

if __name__ == "__main__":
  smile_detect()
