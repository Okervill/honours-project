from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2


# load the face mask detector model from disk
print('loading face mask detector model...')
maskNet = load_model('mask_detector')

# initialize the video stream and allow the camera sensor to warm up
print('starting video stream...')
vs = VideoStream(src=0).start()
 
# loop over the frames from the video stream
while True:
	# grab the frame and resize it to 800px width
	frame = vs.read()
	resizedFrame = cv2.resize(frame, (224, 224))
	resizedFrame = np.reshape(resizedFrame, [1, 224, 224, 3])
	predictions = maskNet.predict(np.array(resizedFrame, dtype='float32'), batch_size=32)
	print(predictions)
 
	(mask, withoutMask) = predictions[0]
	# determine the class label and colour we'll use to draw
	# the bounding box and text
	label = 'Mask' if mask >= 0.8 else 'No Mask'
	colour = (0, 255, 0) if label == 'Mask' else (0, 0, 255)
		
	# include the probability in the label
	probability = max(mask, withoutMask) * 100
	label = f'{label}: {probability:.2f}'
	# display the label and bounding box rectangle on the output
	# frame
	cv2.putText(frame, label, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2)

	# show the output frame
	cv2.imshow('Frame', frame)
	key = cv2.waitKey(1)

	# break on ESC key
	if key == 27:
		break

# do a bit of cleanup
vs.stop()
cv2.destroyAllWindows()
