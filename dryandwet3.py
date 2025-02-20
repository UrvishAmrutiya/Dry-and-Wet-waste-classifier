import cv2
import numpy as np
import tensorflow as tf
import time


model = tf.keras.models.load_model(r"C:\Users\urvis\OneDrive\Desktop\AIML_EL\fine_tuned_model_final.h5")


labels = {0: 'Wet Waste', 1: 'Dry Waste'}
cap = cv2.VideoCapture(0)


frame_buffer_size = 10  # Number of frames to average over
prediction_buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    img = cv2.resize(frame, (300, 300))  
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)  

    # Predict
    pred = model.predict(img)
    prediction = int(pred > 0.5)  

    prediction_buffer.append(prediction)

    
    if len(prediction_buffer) > frame_buffer_size:
        prediction_buffer.pop(0)

    # Calculate the moving average prediction
    stabilized_prediction = np.mean(prediction_buffer)  
    stabilized_label = labels[int(stabilized_prediction > 0.5)]  

    # Display Result
    cv2.putText(frame, f"Prediction: {stabilized_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Waste Classification", frame)

    
    time.sleep(0.1)  

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
