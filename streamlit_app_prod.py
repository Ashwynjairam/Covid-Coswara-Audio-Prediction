import io
import warnings

import librosa
import numpy as np
import streamlit as st
import tensorflow as tf

warnings.filterwarnings('ignore')
# st.write(""" Covid Detection""")

# load the model from disk

# filename = 'C:\\Users\\jairama\\IdeaProjects\\Coswara-Data\\covid_sound_model_1.sav'
# file = open(filename, 'rb')
# print(file)
model = tf.keras.models.load_model('covid_sound_model')


def extract_features(file):
    # try:
    audio, sample_rate = librosa.load(file)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_scaled = np.mean(mfccs.T, axis=0)
    return mfcc_scaled


# except:
# return 'failed'

if __name__ == '__main__':

    st.title('Covid-19 Audio data Detection')

    uploaded_file = st.file_uploader('Upload .wav file')

    if not uploaded_file:
        st.warning("Please upload an audio wav file before proceeding!")
        st.stop()
    else:
        # Decode audio and Predict Right Class
        # audio_sample = uploaded_file.name
        extracted_ft = extract_features(io.BytesIO(uploaded_file.read()))

        if extracted_ft == 'failed':
            st.warning("There was an error reading audio file!")
            st.stop()

        pred = model.predict(extracted_ft.reshape(1, 40))

        reshaped = pred.reshape((1,))
        T = 0.6
        y_pred_bool = reshaped >= T

        st.title('Results')
        if y_pred_bool:
            st.write("This audio is predicted to be of Covid-19 positive person")
        else:
            st.write("This audio is predicted to be of Covid-19 negetive person")
