

from scipy.spatial import distance as dist

from imutils import face_utils
import numpy as np


import dlib
import cv2
import pyautogui, sys, pytweening
print('Press Ctrl-C to quit.')
##----------------!!!chk yoooooooour path data and video-----------#
shape_predictor_src = 'shape_predictor_68_face_landmarks.dat'


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.6
EYE_AR_CONSEC_FRAMES = 3

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")

predictor = dlib.shape_predictor(shape_predictor_src)#edit 1

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]



# loop over frames from the video stream
def blinks(gray ,face):
	status = ''
	rect = face
	if rect is not None:

		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		#leftEye ,rightEye = leftEye_rightEye

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0


		if ear < EYE_AR_THRESH:
			status = 'closed_eye'
			pyautogui.doubleClick()
			pytweening.linear(0.75)
			######### this code to determine x,y
			x, y = pyautogui.position()
			positionStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
			print(positionStr, 'end')
			print('\b' * len(positionStr), 'end')

		# otherwise, the eye aspect ratio is not below the blink
		# threshold
		else:
			status = 'open_eye'
			

	# show the frame
	return status
