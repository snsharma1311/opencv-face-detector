"""
Filename: init.py
Author: Shashank Sharma
"""
import cv2
import numpy as np 

"""
Load pretrained deep learning model for face detection
files are inside directory named 'model'
deploy.prototxt.txt: it is a prototxt file containing deep learning architecture used
res10_300x300_ssd_iter_140000.caffemodel: it is a caffemodel file containing the pre-trained weights
"""
model_prototxt = 'model/deploy.prototxt.txt'
model_weights = 'model/res10_300x300_ssd_iter_140000.caffemodel' 
model = cv2.dnn.readNetFromCaffe(model_prototxt, model_weights)

"""
Initailize videostream either from video file or from a webcam
(comment the video source which is not required)
In this we are taking video file from 'samples' folder
"""
#### 1. From video
cap = cv2.VideoCapture('sample_video/sample.mp4')

#### 2. From webcam (comment previous line and uncomment below line if you are using webcam)
#cap = cv2.VideoCapture(0)

"""
Getting properties from video stream 
"""
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
fps = cap.get(cv2.CAP_PROP_FPS)

"""
Loop over frames and draw boxes over detected faces
press 'q' to quit the application
"""
while(cap.isOpened()):
	ret, frame = cap.read()
	
	### preprocess frame using blobfromimage function
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

	### find faces
	model.setInput(blob)
	faces = model.forward()

	### draw boxes
	for i in range(faces.shape[2]):
		confidence = faces[0,0,i,2]
		if confidence < 0.5:
			continue
		box = faces[0, 0, i, 3:7] * np.array([width, height, width, height])
		(startX, startY, endX, endY) = box.astype("int")
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	cv2.imshow('Face Detector',frame)

	### Press 'q' to quit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
