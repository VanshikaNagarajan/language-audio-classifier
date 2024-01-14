import librosa
import numpy as np
import os
import warnings

class AudioValidator:
    def __init__(self):
        self.invalid_count = 0  # Counter for invalid files

    def validate(self, file_path):
        try:
            # Check if the file size is non-zero
            if os.path.getsize(file_path) == 0:
                print(f"Invalid audio file: {file_path}. Error: zero-size array")
                self.invalid_count += 1
                return False
            # to load the audio files
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio, _ = librosa.load(file_path)

            # Check for clarity and absence of significant noise or distortion
            if np.max(np.abs(audio)) < 0.1:
                print(f"Audio file {file_path} is too noisy or distorted. Skipping.")
                self.invalid_count += 1
                return False



            return True
        except Exception as e:
            print(f"Invalid audio file: {file_path}. Error: {e}")
            self.invalid_count += 1
            return False

    def validate_audio_content(self, audio):
        # Check if the audio is not empty or contains only zeros
        if np.all(audio == 0):
            print(f"Invalid audio file. Error: array contains only zeros")
            self.invalid_count += 1
            return False

        return True

    def get_invalid_count(self):
        return self.invalid_count