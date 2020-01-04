from imutils.video import WebcamVideoStream, FPS
import cv2
import pickle

vs = WebcamVideoStream(src=0).start()
fps = FPS().start()

def load_threshold(PICKLE_PATH):  
	'''
	Load pickle files for contour data, access and load each pickle file for the low/high range
	'''
	with open(PICKLE_PATH + "hsv_low.pickle", 'rb') as f:
		HSV_LOW = pickle.load(f)

	with open(PICKLE_PATH + "hsv_high.pickle", 'rb') as f:
		HSV_HIGH = pickle.load(f)

	return HSV_LOW, HSV_HIGH


while True:
	frame = vs.read()  # Fetch the frame from the camera
	display = frame.copy()

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert into HSV readable

	hsv_low, hsv_high = load_threshold("pickles/")  # Load the pickle files and parse it
	mask = cv2.inRange(hsv, hsv_low, hsv_high)  # Filter HSV range

	cnts,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find the contours
	cnt = max(cnts, key = cv2.contourArea)  # Find the biggest contour
	x,y,w,h = cv2.boundingRect(cnt)  # Get the bounding box of the biggest contour

	# Display the box and the top bisecting point of the box
	cv2.rectangle(display,(x,y),(x+w,y+h),(0,255,0),2)
	cv2.circle(display, (int(x+w/2), y), 5, (255,0,0), thickness=1)

	# Show the frame and the mask
	cv2.imshow('Frame', display)
	cv2.imshow('mask', mask)
	fps.update()  # Update FPS

	k = cv2.waitKey(5) & 0xFF
	if k == 27:  # If ESC is clicked, end the loop 
		break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
