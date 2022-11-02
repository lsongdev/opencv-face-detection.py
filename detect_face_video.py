import os
import cv2

# Load the cascade
# https://github.com/opencv/opencv/tree/master/data
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# To capture video from webcam.
cap = cv2.VideoCapture(0)  # FaceTime Camera
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')
# cap = cv2.VideoCapture("rtsp://admin:lsong940@192.168.8.22/stream1")

# # Read the input image
# img = cv2.imread('test.jpg')

# # Convert into grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Get current width of frame
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
# Get current height of frame
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

FILE_OUTPUT = 'output.avi'

# Checks and deletes the output file
# You cant have a existing file or it will through an error
if os.path.isfile(FILE_OUTPUT):
    os.remove(FILE_OUTPUT)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'X264')
out = cv2.VideoWriter(FILE_OUTPUT, fourcc, 20.0, (int(width), int(height)))

while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    # Saves for video
    out.write(img)

    # Display
    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object
# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
