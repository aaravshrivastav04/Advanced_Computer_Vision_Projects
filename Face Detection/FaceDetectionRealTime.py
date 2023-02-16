import cv2  # Importing OpenCV
import mediapipe as mp  # Importing mediapipe as mp

cap = cv2.VideoCapture(0)

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection()


def label(image, text, x, y, w, h, color):
    cv2.putText(image, text, (x - 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.line(image, (x, y), (x + int(w / 4), y), color, 4)
    cv2.line(image, (x, y), (x, y + int(h / 4)), color, 4)
    cv2.line(image, (x + int(w / 1.5), y), (x + w, y), color, 4)
    cv2.line(image, (x + w, y), (x + w, y + int(h / 4)), color, 4)
    cv2.line(image, (x, y + int(h / 1.5)), (x, y + h), color, 4)
    cv2.line(image, (x, y + h), (x + int(w / 4), y + h), color, 4)
    cv2.line(image, (x + int(w / 1.5), y + h), (x + w, y + h), color, 4)
    cv2.line(image, (x + w, y + int(h * 3 / 4)), (x + w, y + h), color, 4)


while True:

    ret, frame = cap.read()

    if ret:

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(frame_rgb)

        height, width = frame.shape[:2]

        if results.detections is not None:
            for detection_id, detection in enumerate(results.detections):
                relative_bounding_box = detection.location_data.relative_bounding_box

                x = int(relative_bounding_box.xmin * width)
                y = int(relative_bounding_box.ymin * height)
                w = int(relative_bounding_box.width * width)
                h = int(relative_bounding_box.height * height)

                confs = []

                if int(detection.score[0] * 100) > 80:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), -1)
                    alpha = 0.5
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                    label(frame, str(int(detection.score[0] * 100)) + "%", x, y, w, h, (255, 255, 255))
        else:
            pass
    else:
        raise Exception("Unable to read data")

    cv2.imshow("Video", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
