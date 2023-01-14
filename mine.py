import cv2
import os
import numpy as np
from gtts import gTTS
import tensorflow as tf
import time 
import mediapipe as mp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

language = 'en'



label_lines = [line.rstrip() for line in tf.io.gfile.GFile("training_set_labels.txt")]

model = tf.keras.models.load_model("mymodel.h5")

def predict(image_data):
    resized_image = cv2.resize(image_data, (224, 224))
    print (resized_image.shape)
    predictions = model.predict(resized_image)
    
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    max_score = 0.0
    res = ''
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        if score > max_score:
            max_score = score
            res = human_string

    return res, max_score
    

live_stream = cv2.VideoCapture(0)
current_word = ""

def speak_letter(letter):
    # Create the text to be spoken
    prediction_text = letter

    # Create a speech object from text to be spoken
    speech_object = gTTS(text=prediction_text, lang=language, slow=False)

    # Save the speech object in a file called 'prediction.mp3'
    speech_object.save("prediction.mp3")

    # Playing the speech using mpg321
    os.system("afplay prediction.mp3")
    
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                #if id ==0:
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
    
cap.release()
cv2.destroyAllWindows()


    


