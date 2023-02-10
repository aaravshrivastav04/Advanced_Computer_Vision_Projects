import cv2
import imutils

cap = cv2.VideoCapture('Videos/cars.mp4')

vehicle_classifier = cv2.CascadeClassifier('haarcascade_car.xml')

while True:

    ret, frame = cap.read()

    try:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except:
        exit()

    vehicles = vehicle_classifier.detectMultiScale(gray_frame, 1.2, 3)

    for x, y, w, h in vehicles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

    frame = imutils.resize(frame, width=1000, height=1000)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()