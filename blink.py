# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
import tkinter as tk
from imutils import face_utils
import numpy as np
from threading import Thread 
import imutils
import time
import dlib
import cv2
import pyautogui, sys, pytweening
print('Press Ctrl-C to quit.')
##----------------!!!chk yoooooooour path data and video-----------#
shape_predictor_src = 'shape_predictor_68_face_landmarks.dat'
class GuiThread(Thread):
	def __init__(self):
		Thread.__init__(self)
	def run(self):
		write_slogan()
def write_slogan():
	print("Tkinter is easy to use!")

	root = tk.Tk()
	frame = tk.Frame(root)
	frame.pack()
	#img = PhotoImage(file='a.gif')
	button1 = tk.Button(text='well', width=22 , height= 20 , bg='red' , fg='blue' , bd=10 , cursor='heart'   , underline=2 , state='normal' , highlightthickness=55)
	button1.pack(side=tk.LEFT)
	print('button1')
	button2 = tk.Button(text='wellcome', width=22 , height= 20 , bg='red' , fg='blue' , bd=10 , cursor='heart'   , underline=2 , state='normal' , highlightthickness=55)
	button2.pack(side=tk.LEFT)
	print('button2')
	button3 = tk.Button(text='well', width=22 , height= 20 , bg='red' , fg='blue' , bd=10 , cursor='heart'   , underline=2 , state='normal' , highlightthickness=55)
	button3.pack(side=tk.LEFT)
	print('button3')
	button4 = tk.Button(text='wellcome', width=22 , height= 20 , bg='red' , fg='blue' , bd=10 , cursor='heart'   , underline=2 , state='normal' , highlightthickness=55)
	button4.pack(side=tk.LEFT)
	print('button4')

	s = getOffset(frame, allowDebugDisplay=True, trackAverageOffset=True, directInferenceLeftRight=True)
	print(s)
	#root.mainloop()
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
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_src)#edit 1


# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# loop over frames from the video stream
def blinks(gray,faces):
	status = ''
	if True:
		rect = faces
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		print("step 3")
		
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		print("step 4")
		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		print("step 5")
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0


		if ear < EYE_AR_THRESH:
			status = 'closed_eye'

			thread1 = GuiThread()
			thread1.start()
			#write_slogan()
			#print(a)
			pyautogui.click()
			#pyautogui.doubleClick()  # perform a left-button double click
			pytweening.linear(0.75)
			######### this code to determine x,y
			x, y = pyautogui.position()
			positionStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
			print(positionStr, end='')
			print('\b' * len(positionStr), end='', flush=True)
		# otherwise, the eye aspect ratio is not below the blink
		# threshold
		else:
			status = 'open_eye'

			write_slogan()
			#print(a)
	# show the frame
	return status
