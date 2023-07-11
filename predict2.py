from ultralytics import YOLO
import cv2
import time

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
model = YOLO("best.pt")

while True:
    retval, image = video.read()
    if not retval: break
    image = cv2.flip(src=image, flipCode=1)
    #time.sleep(0.5)
    
    predict = model.predict(image)
    image = predict[0].plot()
    cv2.imshow(winname="Video", mat=image)
    if cv2.waitKey(1) == ord("q"): break
    
cv2.destroyAllWindows()