import streamlit as st
import sounddevice as sd
import librosa
import numpy as np
import scipy.io.wavfile as wav
import joblib
import os
import time
import matplotlib.pyplot as plt

# Load the trained model and scaler
model = joblib.load("ser_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define emotion labels
label_mapping = {0: "😊 Happy", 1: "😢 Sad", 2: "😡 Angry"}

# Function to record live audio
def record_audio(filename="live_audio.wav", duration=5, sr=16000):
    with st.spinner("🎙️ Recording... Speak now!"):
        recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        wav.write(filename, sr, (recording * 32767).astype(np.int16))
        time.sleep(1)
    return filename

# Function to extract features from audio
def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr))
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        
        features = np.concatenate((mfccs_mean, mfccs_std, [mel_spectrogram, chroma])).reshape(1, -1)
        return features, y
    except Exception as e:
        st.error(f"❌ Error processing audio: {e}")
        return None, None

# Streamlit UI
st.set_page_config(page_title="TONE TRACK", page_icon="🎤", layout="centered")
st.title("🎤 TONE TRACK - Speech Emotion Recognition")
st.markdown("Record your voice and let AI predict the emotion!")

st.markdown("---")
# Helpful tip before recording
st.info("💡 Tip: Try expressing emotions clearly while recording, like a stage actor 🎭")
# Record & Predict Button
if st.button("🎙️ Record & Predict", help="Click to start recording your voice"):    
    file_path = record_audio()
    features, waveform = extract_features(file_path)
    
    if features is not None:
        st.success("✅ Audio recorded successfully!")
        st.audio(file_path, format='audio/wav')  # Play recorded audio
        
        # Show waveform
        fig, ax = plt.subplots()
        ax.plot(waveform, color='purple')
        ax.set_title("🎵 Recorded Audio Waveform")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

        # Show raw features and scaled features
        st.markdown("### 🧠 Feature Debug Info")
        st.write("📌 Feature shape (before scaling):", features.shape)
        st.write("📌 Raw features:", features)

        try:
            features_scaled = scaler.transform(features)
            st.write("📌 Scaled features:", features_scaled)
        except Exception as e:
            st.error(f"Scaler error: {e}")
            st.stop()

        # Predict emotion
        try:
            prediction = model.predict(features_scaled)[0]
            predicted_emotion = label_mapping.get(prediction, "Unknown")
            st.success(f"**Predicted Emotion: {predicted_emotion}**")
            st.write("🔮 Raw model output (class index):", prediction)
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.error("❌ Failed to process audio. Please try again.")
