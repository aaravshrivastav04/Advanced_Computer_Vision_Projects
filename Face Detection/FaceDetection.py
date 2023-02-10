import cv2  # Importing OpenCV
import mediapipe as mp  # Importing mediapipe as mp
import imutils
import time

cap = cv2.VideoCapture('Videos/Video_3.mp4')

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection()

past_time = 0


def label(image, text, x, y, w, h, color):
    cv2.putText(image, text, (x - 30, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 255), 3)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
    cv2.line(image, (x, y), (x + int(w / 4), y), color, 20)
    cv2.line(image, (x, y), (x, y + int(h / 4)), color, 20)
    cv2.line(image, (x + int(w / 1.5), y), (x + w, y), color, 20)
    cv2.line(image, (x + w, y), (x + w, y + int(h / 4)), color, 20)
    cv2.line(image, (x, y + int(h / 1.5)), (x, y + h), color, 20)
    cv2.line(image, (x, y + h), (x + int(w / 4), y + h), color, 20)
    cv2.line(image, (x + int(w / 1.5), y + h), (x + w, y + h), color, 20)
    cv2.line(image, (x + w, y + int(h * 3 / 4)), (x + w, y + h), color, 20)


while True:

    ret, frame = cap.read()

    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except:
        exit()

    results = face_detection.process(frame_rgb)

    height, width = frame.shape[:2]

    for id, detection in enumerate(results.detections):
        relative_bounding_box = detection.location_data.relative_bounding_box

        x = int(relative_bounding_box.xmin * width)
        y = int(relative_bounding_box.ymin * height)
        w = int(relative_bounding_box.width * width)
        h = int(relative_bounding_box.height * height)

        if int(detection.score[0] * 100) > 60:
            label(frame, str(int(detection.score[0] * 100)) + "%", x, y, w, h, (255, 0, 255))

    current_time = time.time()
    fps = 1 / (current_time - past_time)
    past_time = current_time

    cv2.putText(frame, "FPS: " + str(int(fps)), (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 255, 0), 10)

    frame = imutils.resize(frame, width=1000, height=1000)
    cv2.imshow("Video", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
