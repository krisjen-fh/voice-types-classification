import librosa
import numpy as np   
import os
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import soundfile as sf
import librosa
import numpy as np
from scipy import signal

class PreProcess:
    def __init__(self, file_path, output_dir):
        self.file_path = file_path
        self.output_dir = output_dir
    
    def trimmed_audio(self):
        audio = AudioSegment.from_file(self.file_path)

        # Detect non-silent parts
        nonsilent_ranges = detect_nonsilent(audio, min_silence_len=2000, silence_thresh=-40)

        # Extract the first non-silent range (if any)
        if nonsilent_ranges:
            start_trim = nonsilent_ranges[0][0]
            trimmed_audio = audio[start_trim:]
        else:
            trimmed_audio = audio

        base_name = os.path.basename(self.file_path)  # Get file name
        trimmed_name = base_name.replace(".wav", "_trimmed.wav")
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure output directory exists
        self.output_path_trimmed = os.path.join(self.output_dir, trimmed_name)
        if not os.path.exists(self.output_path_trimmed):
            trimmed_audio.export(self.output_path_trimmed, format="wav")
            print(f'Trimmed audio tersimpan pada: {self.output_path_trimmed}')
        else:
            print(f"File {self.output_path_trimmed} sudah ada, tidak ada file yang ditulis.")
        return self.output_path_trimmed
    
    def filtered_audio_lpf(self, cutoff_freq):
        """
        Apply a high-pass filter to the audio file and save the filtered version.
        """
        try:
            file_path = self.output_path_trimmed
            # Read audio file
            audio_data, frame_rate = sf.read(file_path)
            
            # Ensure the audio data is mono (1 channel)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Design the high-pass filter
            nyquist_rate = frame_rate / 2.0
            normal_cutoff = cutoff_freq / nyquist_rate
            b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
            
            # Apply the filter
            filtered_audio_data = signal.filtfilt(b, a, audio_data)
            
            # Prepare output path
            os.makedirs(self.output_dir, exist_ok=True)  # Ensure the directory exists
            base_name = os.path.basename(file_path)
            filtered_file_name = base_name.replace(".wav", "_lpf.wav")
            self.output_path_filtered = os.path.join(self.output_dir, filtered_file_name)
            
            # Save the filtered audio file
            sf.write(self.output_path_filtered, filtered_audio_data, frame_rate)
            print(f"Filtered audio saved at: {self.output_path_filtered}")
            return self.output_path_filtered
        
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None
        
    def segmented_audio(self, target_sr=44100, max_duration=3):
        """
        Preprocess an audio file: normalize, resample, and segment.
        """
        file_path = self.output_path_filtered
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        print(f"Original audio length: {len(y)} samples, sampling rate: {sr}")

        # Normalize amplitude
        y = librosa.util.normalize(y)

        # Resample to target sampling rate
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        # Check if the audio duration is less than max_duration
        duration = len(y) / target_sr
        if duration <= max_duration:
            print(f"Audio length ({duration:.2f}s) is less than or equal to max_duration. Returning the original file path.")
            return [file_path]

        # Segment audio into clips of max_duration
        samples_per_clip = target_sr * max_duration
        num_clips = int(np.ceil(len(y) / samples_per_clip))

        # Save segmented audio
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        self.audio_segmented = []
        for i in range(num_clips):
            start = i * samples_per_clip
            end = min((i + 1) * samples_per_clip, len(y))
            segment = y[start:end]

            # Save the segment
            segment_name = f"{base_name}_segment_{i+1}.wav"
            segment_path = os.path.join(self.output_dir, segment_name)
            sf.write(segment_path, segment, target_sr)
            self.audio_segmented.append(segment_path)
            print(f"Saved: {segment_path}")
        
        return self.audio_segmented
    
    def get_audio_length(self,file_path):
        # Load the audio file
        audio, sr = librosa.load(file_path)
        audio_length_in_seconds = len(audio) / sr
        
        return audio_length_in_seconds