import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model = load_model('tomato_ripeness_final_model.h5')

# Streamlit app title
st.title("Tomato Ripeness Detection")

# Initialize variables for counting ripe and unripe tomatoes
ripe_count = 0
unripe_count = 0

# Define image preprocessing function
def preprocess_image(frame, img_width=128, img_height=128):
    # Resize the frame to match the input size expected by the model
    frame_resized = cv2.resize(frame, (img_width, img_height))
    # Convert the image to an array and scale it (as done during training)
    img_array = image.img_to_array(frame_resized) / 255.0
    # Add batch dimension (since the model expects batches of images)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# OpenCV video capture from webcam
cap = cv2.VideoCapture(0)

# Streamlit layout
stframe = st.empty()
ripe_placeholder = st.empty()
unripe_placeholder = st.empty()

# Loop for capturing video frames
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        st.write("Camera not found!")
        break

    # Preprocess the frame and predict ripeness
    img_array = preprocess_image(frame)
    prediction = model.predict(img_array)

    # Determine ripeness and draw results on the frame
    if prediction > 0.5:
        label = "Ripe"
        ripe_count += 1
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        label = "Unripe"
        unripe_count += 1
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display frame in Streamlit
    stframe.image(frame, channels="BGR")

    # Update ripe and unripe counts in Streamlit
    ripe_placeholder.write(f"Ripe Tomatoes Count: {ripe_count}")
    unripe_placeholder.write(f"Unripe Tomatoes Count: {unripe_count}")

    # Stop the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
