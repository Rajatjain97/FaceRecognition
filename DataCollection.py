import cv2
import numpy as np

# INIT Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

skip = 0
dataset_path = './face_generated_dataset/'

face_data = [] 

fila_name = input("Enter the name: ")
while True:

	ret,frame = cap.read()
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	
	if ret == False:
		continue

	faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
	if len(faces) == 0:
		continue

	for (x,y,w,h) in faces:
		#Get the face Region of Interest
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		skip += 1
		if skip % 10 == 0: # taking every 10th frame
			face_data.append(face_section)
			print(len(face_data))


	cv2.imshow("faces",frame)
	cv2.imshow("Face_Section",face_section)
	# cv2.imshow("Gray Frame",gray_frame)

	keyPressed = cv2.waitKey(1) & 0xFF
	if keyPressed == ord('q'):
		break

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

np.save(dataset_path+fila_name+'.npy',face_data)
print("Data Successfully Saved at" + dataset_path + fila_name + "npy")

cap.release()
cv2.destroyAllWindows()