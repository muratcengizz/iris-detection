from ultralytics import YOLO
import cv2
import os 
import matplotlib.pyplot as plt
model = YOLO("best.pt")

path = os.chdir("C:/Users/murat/Documents/computer_vision3/iris_detection/test/images")
files = os.listdir(path)


def find_location(predict):
    for result in predict:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            names = result.names[int(box.cls[0])]
    
    return [names, x1, y1, x2, y2]
            

def find_contours(image, x1, y1, x2, y2):
    cropped_image = image[x1-50:y1+100, x2-100:y2+100]
    gray_image = cv2.cvtColor(src=cropped_image, code=cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image=gray_image, threshold1=20, threshold2=100)
    contours, hierarchy = cv2.findContours(image=edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    res = cv2.drawContours(cropped_image, contours, -1, (0, 255, 0), 3)
    image[x1-50:y1+100, x2-100:y2+100] = cropped_image
    return image
    
    
while True:
    for file in files:
        img = cv2.imread(filename=file)
        
        pred = model.predict(img)
        img = pred[0].plot()
        #image = find_contours(image=img, x1=x1, y1=y1, x2=x2, y2=y2)
        cv2.imshow(winname="detection", mat=img)
        #plt.imshow(X=image), plt.show()
        if cv2.waitKey(0) == ord("q"): continue
    


cv2.destroyAllWindows()


