import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_cascade = cv2.CascadeClassifier("haarcascade.xml")
skip = 0
face_data = []
dataset_path = './Data/'
file_name = input('Please Enter Your Name ')

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
    faces = sorted(faces, key=lambda f: f[2]*f[3])

    for face in faces[-1:]:
        x,y,w,h = face
        cv2.rectangle(frame,(x,y), (x+w,y+h), (255,0,0), 2)

        offset = 10
        face_selection = frame[y-offset:y+h+offset, x-offset: x+w+offset]
        face_selection = cv2.resize(face_selection,(100,100))
        skip+=1
        if skip%10 == 0:
            face_data.append(face_selection)
            print(len(face_data))

    cv2.imshow(" Frame", frame)
    cv2.imshow("Frame Section", face_selection)
    # cv2.imshow( "Gray Frame", gray_frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

np.save(dataset_path+file_name+'.npy', face_data)
print("Data saved successfully at "+dataset_path+file_name+'.npy')
cap.release()
cv2.destroyAllWindows()