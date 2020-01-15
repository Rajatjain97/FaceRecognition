import cv2
import numpy as np
import os

########### KNN Code ###########
def distance(v1,v2):
	return np.sqrt(((v1-v2)**2).sum())

def knn(train,test,k = 5):
	dist = []

	for i in range(train.shape[0]):
		# get the vector and label
		ix = train[i,:-1] 
		iy = train[i,-1] # labels
		# compute the distance from test point
		d = distance(test,ix)
		dist.append([d,iy])
	# Sort based on distance and get top k
	dk = sorted(dist, key=lambda x: x[0])[:k]

	# Retrieve only the labels
	labels = np.array(dk)[:, -1]
	
	# Get frequencies of each label
	output = np.unique(labels, return_counts=True)
	# Find max frequency and corresponding label
	index = np.argmax(output[1])
	return output[0][index]
################################

# INIT Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

skip = 0
dataset_path = './face_generated_dataset/'

face_data = [] 
labels = []

class_id = 0 # Labels for the given file
names = {} #Mapping btw id - name

# Data Preparation
for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		print("Loaded " + fx)
		names[class_id] = fx[:-4]
		data_item = np.load(dataset_path+fx)
		face_data.append(data_item)

		# Create Labels for the class
		target = class_id*np.ones((data_item.shape[0],))
		class_id += 1
		labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)

# Testing

while True:
	ret,frame = cap.read()
	if ret == False:
		continue

	faces = face_cascade.detectMultiScale(frame,1.3,5)

	for (x,y,w,h) in faces:

		#Get the ROI
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		# Predicted Label out
		out = knn(trainset,face_section.flatten())

		pred_name = names[int(out)]
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

	cv2.imshow("Frame",frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()