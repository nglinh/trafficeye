import cv2
import numpy as np
import sys, getopt

def main(argv):
	input_file = ''
	output_file = ''
	cascade_file = ''
	try:
		opts, args = getopt.getopt(argv,"hvi:o:c:",["ifile=","ofile=", "video=", "cascade="])
	except getopt.GetoptError:
		print 'test.py [-v] -i <inputfile> -o <outputfile>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'test.py [-v] -i <inputfile> -o <outputfile>'
			sys.exit()
		elif opt in ("-i", "--ifile"):
			input_file = arg
		elif opt in ("-o", "--ofile"):
			output_file = arg
		elif opt in ("-v", "--video"):
			video_mode = True
		elif opt in ("-c", "--cascade"):
			cascade_file = arg
	if video_mode:
		detectVideo(input_file, arg, output_file)
	cv2.destroyAllWindows()

def detectVideo(input_file, cascade_file, output_file=''):
	cap = cv2.VideoCapture(input_file)
	car_cascade = cv2.CascadeClassifier(cascade_file)
	if output_file:
		# Define the codec and create VideoWriter object
		fourcc = cv2.VideoWriter_fourcc(*'FMP4')
		out = cv2.VideoWriter(output_file, fourcc, 14.9999, (640,480))
	scaleDown = 0.16
	scaleUp = 6.25
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret:
			canvas = cv2.resize(frame, None, fx=scaleDown, fy=scaleDown, interpolation=cv2.INTER_CUBIC)
			gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
			cars = car_cascade.detectMultiScale(gray, 1.1, 4, minSize=(20,20), maxSize=(45,45))
			for (x,y,w,h) in cars:
				cv2.rectangle(frame, (int(x*scaleUp), int(y*scaleUp)), (int((x+w)*scaleUp), int((y+h)*scaleUp)), (0,255,0), 2)
			
			cv2.imshow('heeeyyyyy', frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

			if output_file:
				out.write(frame)
	cap.release()

if __name__ == '__main__':
	main(sys.argv[1:])