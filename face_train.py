#!/usr/bin/env python

import os,cv2,pickle 
import numpy as np
from PIL import Image
DIR=os.path.dirname(os.path.abspath(__file__))
img_dir=os.path.join(DIR,"IMG")				#directory of the image 
face_casade=cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

recognizer=cv2.face.LBPHFaceRecognizer_create()

x_train=[]
y_labels=[]
currt_id=0
label_id={}
for root, dirs,files in os.walk(img_dir):
	for file in files:
		if file.endswith("png") or file.endswith("JPG") or file.endswith("jpg") or file.endswith("PNG") or file.endswith("jpeg") :
			path=os.path.join(root,file)
			label=os.path.basename(os.path.dirname(path)).replace(" ","-").lower()	#path or root
			#print(label, path)
			#x_train.append(path)			#show the par
			#y_labels.append(label)
			if not label in label_id:
				label_id[label]=currt_id	
				currt_id+=1
			
			id_=label_id[label]
			#print(label_id)
			pill_img=Image.open(path).convert("L")	#gray scale img
			img_array=np.array(pill_img,"uint8")	#img format
			#print(img_array)
			face=face_casade.detectMultiScale(img_array, scaleFactor=1.5, minNeighbors=5)
			
			for (x,y,w,h) in face:
				roi=img_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)
#print(y_labels)
#print(x_train)

with open("labels.pickle",'wb') as f:			#creating and writing the data to a file
	pickle.dump(label_id, f)

try:
	recognizer.train(x_train, np.array(y_labels))
	recognizer.save("facetrainner.yml")						#face trained data
	print("\n\n\nIMAGE SUCCESSFULLY TRAINED")
except :
	print("\n\n\nADD DIFFERENT TYPE OF FACE ON IMG ATLEAST 2")
