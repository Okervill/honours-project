from imutils.video import VideoStream
import numpy as np
import imutils
import cv2

def detect_faces(frame, faceNet):
	# grab the dimensions of the frame and then construct a blob from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and get faces
	faceNet.setInput(blob)
	detections = faceNet.forward()

	locations = []
	predictions = []

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
				# store face and bounding box
				locations.append((startX, startY, endX, endY))
				predictions.append(confidence)

	# return the faces and their locations
	return (locations, predictions)


# load our serialized face detector model from disk
print('loading face detector model...')
prototxtPath = 'face_detector/deploy.prototxt'
weightsPath = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# initialize the video stream and allow the camera sensor to warm up
print('starting video stream...')
vs = VideoStream(src=0).start()
 
# loop over the frames from the video stream
while True:
	# grab the frame and resize it to 800px width
	frame = vs.read()
	frame = imutils.resize(frame, width=800)

	# detect faces in the frame and determine whether or
	# not they are wearing a face mask
	(locations, predictions) = detect_faces(frame, faceNet)

	# loop through each face detected
	for (box, prediction) in zip(locations, predictions):
		# get the bounding box and prediction
		(startX, startY, endX, endY) = box
		colour = (0, 255, 0)

		# display the bounding box
		cv2.rectangle(frame, (startX, startY), (endX, endY), colour, 2)

	# show the output frame
	cv2.imshow('Frame', frame)
	key = cv2.waitKey(1)

	# break on ESC key
	if key == 27:
		break

# do a bit of cleanup
vs.stop()
cv2.destroyAllWindows()
