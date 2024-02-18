import cvzone
import cv2
import os

# Path to the folder containing test images
folder_path = 'D:/ComputerVisionProj/vehicleClassification/test'

# List all files in the folder with specified image file extensions
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Initialize the classifier using a pre-trained model and labels
myClassifier = cvzone.Classifier('MyModel/keras_model.h5', 'MyModel/labels.txt')

print("""
    Indexes:
    0 - Trucks classification
    1 - Cars classification
    2 - Motor bikes classification
    3 - Three wheels classification
    4 - Bicycles classification
    """)

# Loop through each image in the specified folder
for i in image_files:
    # Read the image from the folder
    img = cv2.imread(os.path.join(folder_path, i))

    # Resize the image for better display
    img = cv2.resize(img, (0, 0), fx=5, fy=5)

    # Get predictions and index from the classifier
    predictions, index = myClassifier.getPrediction(img)
    print(predictions, index)

    # Display the resized image
    cv2.imshow("Image", img)

    # Wait for 2 seconds before moving to the next image
    cv2.waitKey(2000)
