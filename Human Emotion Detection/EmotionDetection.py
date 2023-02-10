import cv2
from deepface import DeepFace

image = cv2.imread("Images/Surprised.jpg")

face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray_image, 1.3, 5)


def label(image, text, x, y, w, h, color):
    cv2.putText(image, text, (x - 25, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
    cv2.line(image, (x, y), (x + int(w / 4), y), color, 4)
    cv2.line(image, (x, y), (x, y + int(h / 4)), color, 4)
    cv2.line(image, (x + int(w / 1.5), y), (x + w, y), color, 4)
    cv2.line(image, (x + w, y), (x + w, y + int(h / 4)), color, 4)
    cv2.line(image, (x, y + int(h / 1.5)), (x, y + h), color, 4)
    cv2.line(image, (x, y + h), (x + int(w / 4), y + h), color, 4)
    cv2.line(image, (x + int(w / 1.5), y + h), (x + w, y + h), color, 4)
    cv2.line(image, (x + w, y + int(h * 3 / 4)), (x + w, y + h), color, 4)


for x, y, w, h in faces:

    face = image[y:y + h, x:x + w]
    results = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
    conf = 0
    for _ in results[0]:
        results = results[0][_]
        break
    for emotion in results:
        if results[emotion] > conf:
            conf = results[emotion]
        else:
            continue
    for emotion in results:
        if results[emotion] == conf:
            label(image, emotion + " " + str(int(conf)) + "%", x, y, w, h, (255, 255, 255))

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
