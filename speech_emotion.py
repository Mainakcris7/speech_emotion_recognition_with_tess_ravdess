import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model

st.write("## Speech emotion recognition 🗣️😡😁😑")
st.write("If the audio file is not in **.wav** format, please visit: https://cloudconvert.com/ to convert")
audio_file = st.file_uploader("Upload the audio file (.wav file)")


@st.cache_resource
def load_the_model():
    return load_model('dense_2.keras')


def get_audio_data(audio_file, n_mfcc=64):
    audio_data, _ = librosa.load(audio_file)
    mfcc_data = librosa.feature.mfcc(y=audio_data, n_mfcc=n_mfcc)
    mfcc_agg = mfcc_data.T.mean(axis=0)
    return np.expand_dims(mfcc_agg, axis=0)


label_dict = {
    0: 'neutral',
    1: 'happy',
    2: 'sad',
    3: 'angry',
    4: 'fear',
    5: 'disgust',
    6: 'surprised'}

if audio_file:
    st.audio(audio_file)
    input_data = get_audio_data(audio_file)
    with st.spinner():
        model = load_the_model()
    if st.button("Predict"):
        pred = model.predict(input_data)
        pred = np.argmax(pred, axis=1)[0]
        if (pred == 0):
            st.write(
                f"### Predicted emotion: :blue[{label_dict[pred].capitalize()}]😐")
        elif (pred == 1):
            st.write(
                f"### Predicted emotion: :blue[{label_dict[pred].capitalize()}]😃")
        elif (pred == 2):
            st.write(
                f"### Predicted emotion: :blue[{label_dict[pred].capitalize()}]😢")
        elif (pred == 3):
            st.write(
                f"### Predicted emotion: :blue[{label_dict[pred].capitalize()}]😡")
        elif (pred == 4):
            st.write(
                f"### Predicted emotion: :blue[{label_dict[pred].capitalize()}]😨")
        elif (pred == 5):
            st.write(
                f"### Predicted emotion: :blue[{label_dict[pred].capitalize()}]🤮")
        elif (pred == 6):
            st.write(
                f"### Predicted emotion: :blue[{label_dict[pred].capitalize()}]😮")
