import cv2
capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    # frame = cv2.Canny(frame, 100, 75)
    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
capture.release()
cv2.destroyAllWindows()
