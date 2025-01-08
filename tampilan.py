from PreProcess import PreProcess
from ExtractFeatures import ExtractFeatures
import numpy as np
import os
import statistics
import streamlit as st
import sounddevice as sd
import numpy as np
import wave
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
from PreProcess import PreProcess
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import load

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
        # Get the file name and save it to a location
        file_name_wav = uploaded_file.name
        with open(file_name_wav, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File berhasil diunggah: {file_name_wav}")

        if st.button("🎶 Analisis Suara"):
            try:
                print("Step 1: Pre-process")
                output_dir_name = './output'
                os.makedirs(output_dir_name, exist_ok=True)

                input_directory = os.path.join(output_dir_name, uploaded_file.name)
                with open(input_directory, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                pp = PreProcess(file_name_wav, output_dir_name)

                # Filtered Audio (LPF with cutoff frequency 3000 Hz)
                filtered_output_path = pp.filtered_audio_lpf(cutoff_freq=3000)
                print(f"Step 2 - Filtered Audio: {filtered_output_path}")

                #Segmented Audio (max duration of 3 seconds per segment)
                segmented_output_paths = pp.segmented_audio(target_sr=44100, max_duration=3)
                print(f"Step 3 - Segmented Audio: {segmented_output_paths}")

                #Audio Length for each segment
                for i in segmented_output_paths:
                    audio_length = pp.get_audio_length(i)
                    print(f"Step 4 - Audio Length: {audio_length:.2f} seconds")

                    if audio_length < 3.00:
                        if os.path.exists(i):
                            try:
                                os.remove(i)
                                segmented_output_paths.remove(i)
                                print(f"Deleted {i} because its length is less than 3 seconds.")
                            except PermissionError:
                                print(f"Permission denied to delete {i}.")
                            except Exception as e:
                                print(f"Error while deleting {i}: {e}")
                        else:
                            print(f"The file {i} does not exist.")
                print("Step 2: Ekstraksi Fitur")
                label_preds = []
                for i in segmented_output_paths:
                    ef = ExtractFeatures(i)
                    new_fhe = ef.extract_fhe()
                    new_sc = ef.extract_sc()
                    new_sb = ef.extract_sb()
                    new_avg_f0 = ef.extract_average_f0()
                    new_mfcc = ef.mfcc()
                    new_dmfccs1_ = ef.delta_mfcc1()
                    new_dmfccs2_ = ef.delta_mfcc2()
                    new_chroma = ef.extract_chroma()
                    new_scon_ = ef.spectral_contrast_range()
                    new_sflat_ = ef.spectral_flatness_range()
                    new_srolof_ = ef.spectral_rolloff_range()

                    # Combine all features into a single list
                    # Extract features from the recorded audio
                    new_features = (
                        [new_fhe, new_sc, new_avg_f0, new_sb]
                        + list(new_mfcc)
                        + list(new_chroma)
                        + list(new_dmfccs1_)
                        + list(new_dmfccs2_)
                        + list(new_scon_)
                        + [new_sflat_, new_srolof_]
                    )

                    # Convert any numpy arrays to lists or scalars to avoid unhashable type error
                    new_features = [f[0] if isinstance(f, np.ndarray) else f for f in new_features]

                    # Instead of appending the entire numpy.ndarray to label_preds, 
                    # we extract the most relevant scalar value or feature
                    new_features = np.array(new_features)  # Convert to numpy array to ensure it's hashable
                    # print(f"new_features: {new_features}")

                    # Standardize features
                    scaler = StandardScaler()
                    file = pd.read_csv("Features.csv")
                    file = file.drop(columns=["Unnamed: 0"])

                    drop_features = [
                        "Jenis_suara",
                        "Path_lengkap",
                        "avg_f0",
                        "mfcc_",
                        "chroma_",
                        "dmfccs1_",
                        "dmfccs2_",
                        "scon_",
                    ] 
                    X = file.drop(columns=drop_features)
                    y = file["Jenis_suara"]

                    # Encode labels
                    y_encoded = []
                    for i in y:
                        if i == "sopran":
                            i = 0
                        elif i == "mezzo_sopran":
                            i = 1
                        elif i == "tenor":
                            i = 2
                        elif i == "baritone":
                            i = 3
                        elif i == "bass":
                            i = 4
                        y_encoded.append(i)

                    # Split data for training
                    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
                    X_train = np.array(X_train)
                    scaler.fit(X_train)

                    # Scale new features
                    new_features_scaled = scaler.transform([new_features])

                    # Load the pre-trained model
                    loaded_model = load("model.joblib")
                    result = loaded_model.predict(new_features_scaled)

                    # Map result to corresponding label
                    label_map = {0: "Sopran", 1: "Mezzo Sopran", 2: "Tenor", 3: "Baritone", 4: "Bass"}
                    predicted_label = label_map.get(result[0], "Tidak Diketahui")

                    label_preds.append(predicted_label)  # Here, we store the result, which should be a scalar value

                print(f"label_preds: {label_preds}")
                # Instead of appending numpy array, append the prediction result (scalar)
                pred_mode = statistics.mode(label_preds)
                print(f"pred_mode: {pred_mode}")



                st.success(f"Prediksi suara Anda adalah: *{predicted_label}*")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat menganalisis suara: {e}")
    except Exception as e:
        st.error(f"Gagal mengunggah file audio: {e}")
