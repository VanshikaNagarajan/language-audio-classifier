import librosa
import numpy as np

class FeatureExtractor:
    def extract_mfccs(self, audio, sample_rate):
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs.T, axis=0)
        return mfccs_scaled_features.tolist()