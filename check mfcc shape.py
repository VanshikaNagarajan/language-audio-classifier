import pandas as pd
import librosa

def check_mfccs(audio_file):
    audio, sample_rate = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    if mfccs.shape[0] != 40:
        mfccs = mfccs.T
    print("Shape of MFCCs:", mfccs.shape)
    return mfccs

def main():
    data = pd.read_csv('processed_data.csv')
    sample_audio_file = data['File'][0]
    mfccs = check_mfccs(sample_audio_file)

if __name__ == "__main__":
    main()
