from PreProcess import PreProcess
from ExtractFeatures import ExtractFeatures
import numpy as np
import os
import statistics
import streamlit as st
import sounddevice as sd
import wave
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import load
import time  # For loading spinner

# Custom CSS for background and font styling
st.markdown(
    """
    <style>
        /* Background color */
        .stAef {
            background-color: #f4f4f4;
        }
        /* Title styling */
        h1 {
            color: #2a7fba;
            text-align: center;
        }
        /* Subheader styling */
        h2, h3, h4 {
            color: #1e4d7b;
        }
        /* Button styling */
        div.stButton > button {
            background-color: #4CAF50;
            color: white;
            border-radius: 12px;
            padding: 10px 24px;
            font-size: 16px;
        }
        div.stButton > button:hover {
            background-color: #45a049;
        }
        /* Success and Error Messages */
        .stAlert {
            font-size: 18px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit UI
st.title("OrdinaryVoice")
st.subheader("Ayo kenali suara mu!")
st.write(
    """
    <p style="color:#4d4d4d; font-size:16px;">
    Upload file rekaman suara Anda untuk dianalisis. 
    Kami akan mendeteksi karakteristik unik suara Anda!
    </p>
    """,
    unsafe_allow_html=True,
)

# Upload audio file (only accept .wav)
uploaded_file = st.file_uploader("Pilih file rekaman suara (hanya .wav)", type=["wav"])

if uploaded_file is not None:
    try:
        # Save the uploaded file temporarily
        file_name_wav = uploaded_file.name
        with open(file_name_wav, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File berhasil diunggah: {file_name_wav}")

        # Play the uploaded audio file
        st.audio(file_name_wav, format="audio/wav")

        if st.button("🎶 Analisis Suara"):
            with st.spinner("🎤 Menganalisis suara Anda..."):
                time.sleep(1)  # Simulate loading time for UI purposes
                
                try:
                    # Pre-process
                    print("Step 1: Pre-process")
                    output_dir_name = './output'
                    os.makedirs(output_dir_name, exist_ok=True)

                    pp = PreProcess(file_name_wav, output_dir_name)

                    # Filtered Audio (LPF with cutoff frequency 3000 Hz)
                    filtered_output_path = pp.filtered_audio_lpf(cutoff_freq=3000)
                    print(f"Step 2 - Filtered Audio: {filtered_output_path}")

                    # Segmented Audio (max duration of 3 seconds per segment)
                    segmented_output_paths = pp.segmented_audio(target_sr=44100, max_duration=3)
                    print(f"Step 3 - Segmented Audio: {segmented_output_paths}")

                    # Remove segments shorter than 3 seconds
                    for i in segmented_output_paths:
                        audio_length = pp.get_audio_length(i)
                        if audio_length < 3.00:
                            if os.path.exists(i):
                                os.remove(i)
                                segmented_output_paths.remove(i)

                    # Extract features
                    print("Step 2: Ekstraksi Fitur")
                    label_preds = []
                    for i in segmented_output_paths:
                        ef = ExtractFeatures(i)
                        new_features = (
                            [ef.extract_fhe(), ef.extract_sc(), ef.extract_average_f0(), ef.extract_sb()]
                            + list(ef.mfcc())
                            + list(ef.extract_chroma())
                            + list(ef.delta_mfcc1())
                            + list(ef.delta_mfcc2())
                            + list(ef.spectral_contrast_range())
                            + [ef.spectral_flatness_range(), ef.spectral_rolloff_range()]
                        )
                        new_features = np.array([f[0] if isinstance(f, np.ndarray) else f for f in new_features])

                        # Standardize features
                        scaler = StandardScaler()
                        file = pd.read_csv("Features.csv").drop(columns=["Unnamed: 0"])
                        X = file.drop(columns=["Jenis_suara"])
                        y = file["Jenis_suara"].map({
                            "sopran": 0, "mezzo_sopran": 1, "tenor": 2, "baritone": 3, "bass": 4
                        })
                        X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
                        scaler.fit(X_train)

                        new_features_scaled = scaler.transform([new_features])

                        # Predict using pre-trained model
                        loaded_model = load("model.joblib")
                        result = loaded_model.predict(new_features_scaled)
                        label_map = {0: "Sopran", 1: "Mezzo Sopran", 2: "Tenor", 3: "Baritone", 4: "Bass"}
                        predicted_label = label_map.get(result[0], "Tidak Diketahui")
                        label_preds.append(predicted_label)

                    pred_mode = statistics.mode(label_preds)

                    st.success(f"Prediksi suara Anda adalah: *{pred_mode}*")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat menganalisis suara: {e}")

    except Exception as e:
        st.error(f"Gagal mengunggah file audio: {e}")
