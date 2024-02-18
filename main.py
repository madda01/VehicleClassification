import cvzone
import cv2
import os

folder_path = 'D:/ComputerVisionProj/vehicleClassification/test'
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

capture = cv2.VideoCapture(0)

myClassifier = cvzone.Classifier('MyModel/keras_model.h5', 'MyModel/labels.txt')

print("""
    Indexes:
    0 - Trucks classification
    1 - Cars classification
    2 - Motor bikes classification
    3 - Three wheels classification
    4 - Bicycles classification
    """)

for i in image_files:
    img = cv2.imread(os.path.join(folder_path, i))

    img = cv2.resize(img, (0, 0), fx=5, fy=5)

    predictions, index = myClassifier.getPrediction(img)
    print(predictions, index)

    cv2.imshow("Image", img)
    cv2.waitKey(2000)
