import cv2
from deepface import DeepFace

# start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()

    try:
        # analyze emotion
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # get dominant emotion
        emotion = result[0]['dominant_emotion']

        # display emotion on screen
        cv2.putText(frame,
                    emotion,
                    (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

    except:
        pass

    # show webcam window
    cv2.imshow("Emotion Detection", frame)

    # press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()