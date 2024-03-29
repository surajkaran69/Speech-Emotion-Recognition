import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
from pydub import AudioSegment
import io
import os

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

# Define the get_file_ext function
def get_file_ext(file_name):
    root, ext = os.path.splitext(file_name)
    return ext[1:]

# Define the Streamlit app
def app():
    st.title('Speech Recognition App')
    # Add a file uploader to let users upload the audio file
    uploaded_file = st.file_uploader('Upload an audio file', type=['wav','mp3'])
    if uploaded_file is not None:
        # Convert the audio file to a playable format
        audio_bytes = uploaded_file.read()
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=get_file_ext(uploaded_file.name))
        audio_wav = audio_segment.export(format="wav")
        audio_file = io.BytesIO(audio_wav.read())

        # Display the audio player
        st.audio(audio_bytes)

        # Call the recognize_speech function to recognize the speech in the uploaded file
        predicted_label = recognize_speech(audio_file)
        # Show the predicted label to the user
        st.write('Predicted label:', predicted_label)

# Run the app
if __name__ == '__main__':
    app()

