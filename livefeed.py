import cv2
import tensorflow as tf
import numpy as np

CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

def prepare(filepath):
    IMG_SIZE = 48
    new_array = cv2.resize(filepath, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# model = tf.keras.models.load_model("asl-sign-language-model-20210326-MNIST.h5")
# model = tf.keras.models.load_model("20210331/full.model")

'''
cap = cv2.VideoCapture(0)#use 0 if using inbuilt webcam, use 1 for plugin

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame1 = cv2.resize(frame, (200, 200))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    prediction = model.predict([prepare(gray)])
    final = (CATEGORIES[int(np.argmax(prediction[0]))])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,final,(200,100), font, 1, (0,0,0), 2, cv2.LINE_AA)

    cv2.imshow('Input', frame)


    c = cv2.waitKey(1)
    if c == 27: # hit esc key to stop
        break

cap.release()
cv2.destroyAllWindows()
'''

def show_webcam(model, mirror=False ):
    scale=25

    cam = cv2.VideoCapture(0)
    while True:
        ret_val, image = cam.read()
        if mirror:
            image = cv2.flip(image, 1)


        #get the webcam size
        height, width, channels = image.shape

        #prepare the crop
        centerX,centerY=int(height/2),int(width/2)
        radiusX,radiusY= int(scale*height/100),int(scale*width/100)

        minX,maxX=centerX-radiusX,centerX+radiusX
        minY,maxY=centerY-radiusY,centerY+radiusY

        cropped = image[minX:maxX, minY:maxY]
        resized_cropped = cv2.resize(cropped, (width, height))

        cv2.imshow('my webcam', resized_cropped)
        frame = resized_cropped
        #frame1 = cv2.resize(frame, (200, 200))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        prediction = model.predict([prepare(gray)])
        final = (CATEGORIES[int(np.argmax(prediction[0]))])

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,final,(200,100), font, 1, (0,0,0), 2, cv2.LINE_AA)

        cv2.imshow('Input', frame)


        if cv2.waitKey(1) == 27:
            break  # esc to quit

        #add + or - 5 % to zoom
        if cv2.waitKey(1) == 0:
            scale += 5  # +5

        if cv2.waitKey(1) == 1:
            scale = 5  # +5
    cap.release()
    cv2.destroyAllWindows()


def main():
    model = tf.keras.models.load_model("20210331/full.model")
    show_webcam(model, mirror=True)


if __name__ == '__main__':
    main()
