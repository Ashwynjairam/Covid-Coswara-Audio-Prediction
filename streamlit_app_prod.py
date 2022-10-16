import io
import warnings

import librosa
import numpy as np
import streamlit as st
import tensorflow as tf

warnings.filterwarnings('ignore')

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
    st.markdown("This application detects the Covid-19 infection in a person based on the provided speech sample.")
    st.markdown("The speech sample can be of ***breathing, coughing, counting numbers 0-9 or reciting vowels***.")

    uploaded_file = st.file_uploader("")
    if not uploaded_file:
        st.warning("Please upload .wav file with person's speech or cough sounds before proceeding!")
        st.stop()
    else:
        with st.spinner('Reading the sample...'):
            # Decode audio and Predict Right Class
            # audio_sample = uploaded_file.name
            audio_bytes = io.BytesIO(uploaded_file.read())
            st.audio(audio_bytes)
            extracted_ft = extract_features(audio_bytes)

            if extracted_ft == 'failed':
                st.warning("There was an error reading audio file!")
                st.stop()

        with st.spinner('Analysing the sample...'):
            pred = model.predict(extracted_ft.reshape(1, 40))

            reshaped = pred.reshape((1,))
            T = 0.6
            y_pred_bool = reshaped >= T

        st.header('Result')

        pred_str = "positive" if y_pred_bool else "negative"
        pred_acc = reshaped[0] if y_pred_bool else 1 - reshaped[0]

        pred_acc_percent = int(pred_acc * 100)
        st.success(
            "This sample is predicted to be of **Covid-19 " + pred_str + "** person with " + str(
                pred_acc_percent) + "% accuracy.")
