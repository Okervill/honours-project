from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2

def detectFaces(frame, faceNet):
    # grab the dimensions of the frame and then construct a blob from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and get detect faces
	faceNet.setInput(blob)
	detections = faceNet.forward()

	locations = []

	# loop through the detections
	for i in range(0, detections.shape[2]):
		# get the confidence
		confidence = detections[0, 0, i, 2]

		# confidence should be at least 50%
		if confidence > 0.5:
			# get x,y coordinates for the bounding box
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
   
			face = frame[startY:endY, startX:endX]
			if face.any():
				locations.append((startX, startY, endX, endY))
    
	# return the location of the faces
	return (locations)

def detectMasks(frame, locations, maskNet):
	faces = []
    
	for location in locations:
		# get the section of the frame containing the face
		(startX, startY, endX, endY) = location
		face = frame[startY:endY, startX:endX]
  
		# pre-process the face and store to array
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		faces.append(face)

	# run model against all faces in image in one batch
	faces = np.array(faces, dtype='float32')
	predictions = maskNet.predict(faces, batch_size=32)
 
	return (locations, predictions)

	
if __name__ == "__main__":
    
    # load our serialized face detector model from disk
	print('loading face detector model...')
	prototxtPath = 'face_detector/deploy.prototxt'
	weightsPath = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'
	faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

	# load the face mask detector model from disk
	print('loading face mask detector model...')
	maskNet = load_model('mask_detector')

	# initialize the video stream and allow the camera sensor to warm up
	print('starting video stream...')
	vs = VideoStream(src=3).start()
	
	# loop over the frames from the video stream
	while True:
		frame = vs.read()

		# detect faces in the frame and determine whether or
		# not they are wearing a face mask
		locations = detectFaces(frame, faceNet)
		if len(locations) != 0:
			(locations, predictions) = detectMasks(frame, locations, maskNet)

			# loop through each face detected
			for (box, prediction) in zip(locations, predictions):
				# get the bounding box and prediction
				(startX, startY, endX, endY) = box
				(mask, withoutMask) = prediction

				# determine the class label and colour we'll use to draw
				# the bounding box and text
				label = 'Mask' if mask >= 0.8 else 'No Mask'
				colour = (0, 255, 0) if label == 'Mask' else (0, 0, 255)

				# include the probability in the label
				probability = max(mask, withoutMask) * 100
				label = f'{label}: {probability:.1f}%'

				# display the label and bounding box rectangle on the output
				# frame
				cv2.putText(frame, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY), colour, 4)

		# show the output frame
		cv2.imshow('Frame', frame)
		key = cv2.waitKey(1)

		# break on ESC key
		if key == 27:
			break

	# do a bit of cleanup
	vs.stop()
	cv2.destroyAllWindows()