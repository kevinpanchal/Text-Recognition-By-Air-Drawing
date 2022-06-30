from flask import Flask, render_template, Response, request
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import sys
import os

app = Flask(__name__)
camera = cv2.VideoCapture(0)

@app.route('/')
def main():
	return render_template('index.html')


def auc(y_true, y_pred):
	auc = tf.metrics.auc(y_true, y_pred)[1]
	keras.backend.get_session().run(tf.local_variables_initializer())
	return auc

model = load_model('model_hand.h5', custom_objects={'auc': auc})
def gen_frames():  
	# count the number of points on frame
	counter = 0
	# variable to store all the written letters
	st=""
	anss=""
	success, old_frame = camera.read()
	mask1 = np.zeros_like(old_frame)
	while True:
		success, frame = camera.read()  # read the camera frame
		if not success:
			break
		else:
			font = cv2.FONT_HERSHEY_SIMPLEX
			# hsv in order to detect the object, one can use different value too
			hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV);
			# upper_bound and lower_bound of the hsv, just to identify the object clearly
			lb = np.array([0, 216, 136])
			ub = np.array([255, 255, 255])
			mask = cv2.inRange(hsv, lb, ub)
			
			# bitwise_and operation of the frame and frame, where mask is given
			res = cv2.bitwise_and(frame, frame, mask=mask)
			
			# find the countours
			edged = cv2.Canny(res, 30, 200)
			contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
			img = cv2.drawContours(frame, contours, 0, (0, 255, 0), 0)		
			
			flag=0
			goOn=0
			# check for all the contours, to find the one we need
			for i in contours:			
				# flag is set to 1, just in case when we detect contour
				flag=1
				
				# make the countour of rectangular shape
				(x, y, w, h)=cv2.boundingRect(i)
				# if the area of rectangle is less than 39 sq px, we don't require that contour
				if(cv2.contourArea(i)<39):
					continue;
				cnt = i
				
				# in order to find the center of rectangle
				M = cv2.moments(cnt)
				if M['m00'] != 0:
					cx = int(M['m10'] / M['m00'])
					cy = int(M['m01'] / M['m00'])
					
					# draw the point where center is detected, and each time join previous point with current center
					if counter == 0:
						cx1 = cx
						cy1 = cy
					if counter == 0:
						counter += 1
						img = cv2.drawContours(frame, contours, 0, (0, 255, 0), 0)
					if(cx>=0 and cx<=130 and cy>=0 and cy<=50):
						# case when one wants to clear everything written on screen
						goOn=1
						break
					# mask1 to store the shape written by joining all points
					mask1 = cv2.line(mask1, (cx, cy), (cx1, cy1), (0.0, 255), 2)
					cx1 = cx
					cy1 = cy
				
				# do the addition operation of frame and mask1
				img = cv2.add(frame, mask1)
				
			img = cv2.add(frame, mask1)
			
			# flip is needed because everything we write is the mirror image
			img=cv2.flip(img, 1)
			cv2.putText(img, 'Predict and Clear', (480, 20), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
			cv2.rectangle(img, (480, 430), (640, 460), (50, 50, 50), -1)
			cv2.putText(img, st, (500, 455), font, 1.0, (255, 255, 255), 1, cv2.LINE_AA)
			if(flag==0):
				# when there is no contour detected, then make counter as null
				counter=0
			
			mask2=cv2.flip(mask1, 1)
			if(goOn==1):
				gray = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
				edged = cv2.Canny(gray, 30, 200)

				contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
				ct=0
				crop=[]
				seq=[]
				j=0
				mx=0
				for c in contours:
					x, y, w, h = cv2.boundingRect(c)
					if(w*h>19000):
						crop.append(mask2[y-20:y+h+20, x-20:x+w+20])
						seq.append([x, j])
						j+=1
				seq.sort()
				anss=""
				inside=0
				for ii in range(j):
					inside=1
					# convert the color img to gray scale of mask2
					g=cv2.cvtColor(crop[seq[ii][1]], cv2.COLOR_BGR2GRAY)
					
					# resize the width and height to 28*28, because the model is trained with this size
					rr=cv2.resize(g, (28, 28), interpolation=cv2.INTER_AREA)
					
					# threshold the resized image
					_2, img_thresh2 = cv2.threshold(rr, 0, 255, cv2.THRESH_BINARY)
					new=np.array(img_thresh2).reshape(-1, 28, 28, 1)
					
					# now that's it, just predict
					p2=model.predict(new)
					
					chara2=np.argmax(p2)+ord('A')
					#print(chr(chara2))
					anss=anss+chr(chara2)
				goOn=0
				if(inside!=0):	
					st=anss
					cv2.putText(img, st, (500, 455), font, 1.0, (255, 255, 255), 1, cv2.LINE_AA)
					# for clearing the screen
				mask1 = np.zeros_like(old_frame)
				goOn=0
		
			frame=img
			ret, buffer = cv2.imencode('.jpg', frame)
			frame = buffer.tobytes()
			yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
	return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
	
@app.route('/hand2Text')
def hand2Text():
	return render_template('hand2Text.html')

@app.route('/text2Hand')
def text2Hand():
	return render_template('text2Hand.html')
	
