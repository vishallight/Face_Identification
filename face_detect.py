import numpy as np
import cv2,pickle,os
import webbrowser,pyttsx

cam = cv2.VideoCapture(0)
face_casade=cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("facetrainner.yml")

	
	
label={"person_name":1}
with open("labels.pickle",'rb') as f:
	orgin_label=pickle.load(f)
	label= {v:k for k,v in orgin_label.items()}
print("\n\n \t \t PRESS 'q' TO EXIT")
while 1:
	a,frame=cam.read()
	
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	face=face_casade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	for (x,y,w,h) in face:
		print(x,y,w,h)
		roi_gray=gray[y:y+h, x:x+h]
		
		id_, conf= recognizer.predict(roi_gray)
		if conf>=45 and conf<=85:
			print(id_)
			print(label[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = label[id_]
			
			color=(255,255,255)
			stroke=2
			cv2.putText(frame, name, (x,y), font,1, color, stroke, cv2.LINE_AA)
			
		roi_col=frame#[y:y+h,x:x+w]				#now it capture the full frame when the face is detected
		img="IMG/img_"+str(x)+".JPG"
		#cv2.imwrite(img,roi_col)		#to capture the face 
		color=(0,0,255)
		stroke=2
		width= x + w	
		height=y+h
		
		cv2.rectangle(frame,(x,y),(width,height),color,stroke)
	cv2.imshow('STREAM',frame)
	
	
	if cv2.waitKey(1) & 0xFF==ord('q'):
		break
cam.release()
cv2.destroyAllWindows()
