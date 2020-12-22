import numpy as np
import os
import cv2

def distance(v1,v2):
    return np.sqrt(((v1-v2)**2).sum())

def knn(train,test,k=5):
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d,iy])
        dk = sorted(dist, key=lambda x: x[0])[:k]

        labels = np.array(dk)[:, -1]
        output = np.unique(labels, return_counts=True)
        index = np.argmax(output[1])
        return output[0][index]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

face_cascade = cv2.CascadeClassifier("haarcascade.xml")

skip = 0
face_data = []
labels = []
dataset_path = './Data/'
class_id = 0
names = {}

# Data Preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        # Mapping
        names[class_id] = fx[:-4]
        print("Loaded "+ fx)
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)
        # Labels
        target = class_id*np.ones((data_item.shape[0],))
        class_id +=1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

print(face_dataset.shape)
print((face_labels.shape))

train_dataset = np.concatenate((face_dataset, face_labels), axis=1)
print(train_dataset.shape)

# Testing
while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    # faces = sorted(faces, key=lambda f: f[2]*f[3])

    for face in faces:
        x,y,w,h = face

        offset = 10
        face_selection = frame[y-offset:y+h+offset, x-offset: x+w+offset]
        face_selection = cv2.resize(face_selection,(100,100))

        out = knn(train_dataset, face_selection.flatten())

        pred_name = names[int(out)]
        cv2.putText(frame, pred_name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


    cv2.imshow(" Faces", frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()