from flask import Flask, request, render_template
import numpy as np
import pickle
import joblib
from joblib import load
from tensorflow.keras.preprocessing.sequence import pad_sequences 
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__, template_folder='templates')

@app.route('/') 
def web():
    return render_template('web.html')

@app.route('/predict', methods=["POST"])
def predict():
    tweet = request.form['tweet']
    sample_data = [tweet]

    token = load('tok')
    sequences = token.texts_to_sequences(sample_data)
    data = pad_sequences(sequences,maxlen=24)

    model = tf.keras.models.load_model('./models/')
    result_prediction = model.predict(data)

    probab = round(result_prediction[0][0]*100,2)
    result = str(probab)

    if result_prediction >= 0.55:
        predicted_class = 'Hate Speech'
    else:
        predicted_class = 'Non-Hate Speech'
        
    
    return render_template('web.html', prediction_text='Mood of tweet: {}'.format(predicted_class)) 
   

if __name__=="__main__":
    app.run(port=5555, debug=True)