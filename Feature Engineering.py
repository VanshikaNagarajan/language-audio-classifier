import os
import random
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
import librosa.display
import librosa
from IPython.display import Audio, display
import pandas as pd
import numpy as np



root_dataset_folder = '/Users/vanshika/PycharmProjects/Language Classification/Language Detection Dataset'
folder_paths = [os.path.join(root_dataset_folder, folder) for folder in os.listdir(root_dataset_folder) if os.path.isdir(os.path.join(root_dataset_folder, folder))]
data_list = []


for folder_path in Path(root_dataset_folder).iterdir():
    if folder_path.is_dir():
        language = folder_path.name
        audio_files = glob(os.path.join(folder_path, '*.mp3'))
        random_audio_file = random.choice(audio_files)
        # print(language)
        # using librosa to normalize by default
        audio, sample_rate = librosa.load(random_audio_file, sr=None)

        # display(Audio(filename=random_audio_file))
        plt.Figure(figsize = (12,6))
        librosa.display.waveshow(audio,sr = librosa.get_samplerate(random_audio_file), color = 'b')
        plt.title(f'Wave plot - {language}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        # showing waveforms
        plt.show()


        print(audio)
        print(sample_rate)
        print('Done')
        # for audio_file in audio_files:
        #     audio, _ = librosa.load(random_audio_file, sr=None)
        #     data_list.append({'File': audio_file, 'Language': language})

        # Convert the list of dictionaries to a DataFrame
# df = pd.DataFrame(data_list)
# print(df.head())


print(f"Mean: {np.mean(audio)}")
print(f"Standard Deviation: {np.std(audio)}")

hindi = os.path.join(root_dataset_folder, 'hindi')
marathi = os.path.join(root_dataset_folder, 'marathi')
tamil = os.path.join(root_dataset_folder, 'tamil')
telugu = os.path.join(root_dataset_folder, 'telugu')
bengali = os.path.join(root_dataset_folder, 'bengali')
gujarati = os.path.join(root_dataset_folder, 'gujarati')
kannada = os.path.join(root_dataset_folder, 'kannada')
malayalam = os.path.join(root_dataset_folder, 'malayalam')
punjabi = os.path.join(root_dataset_folder, 'punjabi')
urdu = os.path.join(root_dataset_folder, 'urdu')

# Lists to store file_name, language, and file_path
file_name = []
language = []
file_path = []

train_size0 = 5000
train_size1 = int(train_size0 * 0.2)

# Function to add files to the lists
def add_files(folder, lang, size):
    files = os.listdir(folder)[:size]
    for filename in files:
        file_name.append(filename[:-4])
        language.append(lang)
        file_path.append(os.path.join(folder, filename))

# Add files for each language
add_files(hindi, 'hindi', train_size0)
add_files(marathi, 'marathi', train_size1)
add_files(tamil, 'tamil', train_size1)
add_files(telugu, 'telugu', train_size1)
add_files(bengali, 'bengali', train_size1)
add_files(gujarati, 'gujarati', train_size1)
add_files(kannada, 'kannada', train_size1)
add_files(malayalam, 'malayalam', train_size1)
add_files(punjabi, 'punjabi', train_size1)
add_files(urdu, 'urdu', train_size1)

df = pd.DataFrame({'File': file_path, 'Language': language})
print(df)


print(f"Number of files: {len(file_name)}, Number of language labels:{len(language)}, Length of file_paths: {len(file_path)}")

csv_file_path = '/Users/vanshika/PycharmProjects/Language Classification/Audio Dataset.csv'
df.to_csv(csv_file_path, index = False)
print(f"CSV file saved to: {csv_file_path}")
