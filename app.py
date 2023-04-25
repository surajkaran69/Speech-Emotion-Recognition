import streamlit as st
import tensorflow as tf
import librosa
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('speechemotionrecognition.h5')

# Define the function to recognize speech
def recognize_speech(audio_file):
    # Load the audio file using librosa
    audio, sr = librosa.load(audio_file, duration=3, offset=0.5)

    # Reshape the audio file to match the model input shape
    audio = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    audio = audio.reshape(1, -1)
    
    # Use the model to recognize the speech
    predicted_label = model.predict(audio)
    predicted_label = np.argmax(predicted_label, axis=1)[0]
    
    # Map the predicted label to the corresponding emotion string
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Surprise', 'Sad']
    predicted_emotion = emotions[predicted_label]
    
    return f'Speech Emotion is {predicted_emotion}'

# Define the Streamlit app
def app():
    st.title('Speech Emotion Recognition App')
    st.sidebar.markdown('Connect with me on LinkedIn:')
    st.sidebar.markdown('[My LinkedIn ID](https://www.linkedin.com/in/surajkaran/)')
    
    # Add a file uploader to let users upload the audio file
    uploaded_file = st.file_uploader('Upload an audio file', type=['wav','mp3'])
    
    if uploaded_file is not None:
        # Call the recognize_speech function to recognize the speech in the uploaded file
        predicted_label = recognize_speech(uploaded_file)
        # Show the predicted label to the user
        st.write(predicted_label)

# Run the app
if __name__ == '__main__':
    app()
