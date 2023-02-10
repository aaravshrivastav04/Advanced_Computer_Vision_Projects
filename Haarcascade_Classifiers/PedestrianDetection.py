import cv2
import imutils

cap = cv2.VideoCapture('Videos/walking.mp4')

body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

while True:

    ret, frame = cap.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    pedestrians = body_classifier.detectMultiScale(gray_frame, 1.2, 3)

    for x, y, w, h in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    # frame = imutils.resize(frame, width=1000, height=1000)

    cv2.imshow('Frame', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()