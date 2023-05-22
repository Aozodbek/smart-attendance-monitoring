import cv2
import face_recognition


imgTest = face_recognition.load_image_file('TestImages/Jeff.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
imgCompare = face_recognition.load_image_file('CompareImages/Jeff.jpeg')
imgCompare = cv2.cvtColor(imgCompare,cv2.COLOR_BGR2RGB)


faceLoc = face_recognition.face_locations(imgTest)[0]
encodeDemo = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLoc[3],faceLoc[0]), (faceLoc[1],faceLoc[2]),(0,255,0),(4))

faceLocTest = face_recognition.face_locations(imgCompare)[0]
encodeTest = face_recognition.face_encodings(imgCompare)[0]
cv2.rectangle(imgCompare, (faceLocTest[3],faceLocTest[0]), (faceLocTest[1],faceLocTest[2]),(0,255,0),(4))

results = face_recognition.compare_faces([encodeDemo],encodeTest)
faceDis = face_recognition.face_distance([encodeDemo],encodeTest)
print(results,faceDis)
cv2.putText(imgCompare,f'{results} {round(faceDis[0],2)}', (50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)

cv2.imshow('TestImage', imgTest)
cv2.imshow('CompareImage', imgCompare)
cv2.waitKey(0)


