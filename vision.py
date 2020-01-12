import cv2, pickle, argparse
from imutils.video import WebcamVideoStream, FPS
from networktables import NetworkTables

# Variable constants
ROBORIO_IP = "10.6.12.2"
OUTLIER = -99999

def load_threshold(PICKLE_PATH):  
	'''
	Load pickle files for contour data, access and load each pickle file for the low/high range
	'''
	with open(PICKLE_PATH + "hsv_low.pickle", 'rb') as f:
		HSV_LOW = pickle.load(f)

	with open(PICKLE_PATH + "hsv_high.pickle", 'rb') as f:
		HSV_HIGH = pickle.load(f)

	return HSV_LOW, HSV_HIGH


def create_arguments():
	'''
	Create argpars structure for passing arguments through command line
	'''
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--display", type=int, default=-1, help="Whether or not frames should be displayed")
	ap.add_argument("-t", "--table", type=str, required=True, help="Determine the name of the NetworkTable to push to")
	ap.add_argument("-a", "--area", type=int, default=0, help="Area limit for contour detection")
	ap.add_argument("-f", "--fov", type=int, default=60, help="Degree FOV of camera")
	args = vars(ap.parse_args())
	return args


def main():
	'''
	Main loop for vision capturing
	'''
	vs = WebcamVideoStream(src=0).start()
	fps = FPS().start()

	args = create_arguments()

	NetworkTables.initialize(server=ROBORIO_IP)  # Initialize NetworkTable server
	sd = NetworkTables.getTable(args["table"])  # Fetch the NetworkTable table

	while True:
		frame = vs.read()  # Fetch the frame from the camera
		display = frame.copy()

		height, width = frame.shape[:2]

		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert into HSV readable

		hsv_low, hsv_high = load_threshold("pickles/")  # Load the pickle files and parse it
		mask = cv2.inRange(hsv, hsv_low, hsv_high)  # Filter HSV range

		cnts,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find the contours

		if cnts:  # Prevent error if no contours found
			cnt = max(cnts, key = cv2.contourArea)  # Find the biggest contour
			if cv2.contourArea(cnt) > args['area']:  # Only pass contour if greater than the area limit
				x,y,w,h = cv2.boundingRect(cnt)  # Get the bounding box of the biggest contour
				pixel_offset = (x+w/2)-(width/2)  # tx is horizontal offset
				tx = pixel_offset * (args['fov'] / (width / 2))  # Convert pixel offset to angular offset

				sd.putNumber("tx", tx)  # Push data to table
				
				if args['display'] > 0:  # Only display frames if true
					# Display drawings on frame
					cv2.rectangle(display,(x,y),(x+w,y+h),(0,255,0),2)
					cv2.circle(display, (int(x+w/2), y), 5, (255,0,0), thickness=1)
					cv2.line(display, (int(width/2), 0), (int(width/2), height), (0,0,255), 4)
			else:
				sd.putNumber("tx", OUTLIER)  # Push outlier value to table (not found)
		else:
				sd.putNumber("tx", OUTLIER)  # Push outlier value to table (not found)

		if args['display'] > 0:  # Only display frames if true	
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

	cv2.destroyAllWindows()
	vs.stop()


if __name__ == "__main__":
	main()