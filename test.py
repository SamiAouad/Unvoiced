import tensorflow as tf
import numpy as np
import cv2
import copy
from cvzone.HandTrackingModule import HandDetector

from tensorflow.keras.preprocessing import image as tf_image

LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W'
          'X', 'Y', 'Z', 'del', 'nothing', 'space']

def predict(image):
    temp = copy.deepcopy(image)
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    temp = cv2.resize(temp, (64, 64))
    temp = tf_image.img_to_array(temp)
    temp = np.expand_dims(temp, axis=0)
    temp = tf.keras.applications.vgg19.preprocess_input(temp)

    model = tf.keras.models.load_model("my_model.h5")
    prediction = model.predict(temp)
    label = LABELS[np.argmax(prediction)]
    return label


OFFSET = 20

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

while True:
    success, img = cap.read()
    image = copy.deepcopy(img)
    hands, img = detector.findHands(img)
    try:
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            image_cropped = image[y-OFFSET:y+h+OFFSET, x-OFFSET:x+w+OFFSET]
            print(predict(image_cropped))
            cv2.rectangle(image, (x-OFFSET, y+h+OFFSET), (x+w+OFFSET, y-OFFSET), (0, 255, 0), 2)
            cv2.imshow('image cropped', image_cropped)
        cv2.imshow("Image", image)
    except:
        print("error")
    cv2.waitKey(1)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()



