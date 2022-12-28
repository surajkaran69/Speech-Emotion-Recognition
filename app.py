# Import libraries and modules

from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
import librosa
import os


app = Flask(__name__)

# Loading crop recommendation model

model = load_model('speechemotionrecognition.h5')



# render home page

@ app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    
    if request.method == 'POST':
        f = request.files["file"]
        y, sr = librosa.load(f, duration = 3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        mfcc = mfcc.reshape(1, -1)

        prediction = model.predict([mfcc])
        prediction=np.argmax(prediction,axis=1)
        output = prediction[0]
        
        if output == 0:
            return render_template('index.html', prediction='Speech Emotion is Angry')
        
        elif output == 1:
            return render_template('index.html', prediction='Speech Emotion is Disgust')
        
        elif output == 2:
            return render_template('index.html', prediction='Speech Emotion is Fear')
        
        elif output == 3:
            return render_template('index.html', prediction='Speech Emotion is Happy')
        
        elif output == 4:
            return render_template('index.html', prediction='Speech Emotion is Neutral')
        
        elif output == 5:
            return render_template('index.html', prediction='Speech Emotion is Surprise')
        
        elif output == 6:
            return render_template('index.html', prediction='Speech Emotion is Sad')
   
    


if __name__ == '__main__':
    app.run(debug=True)