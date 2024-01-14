import os
import glob
from pathlib import Path
import pandas as pd
import librosa
from validator import AudioValidator
from FeatureExtractor import FeatureExtractor
import warnings


class AudioProcessor:
    def __init__(self, root_dataset_folder, validator, feature_extractor):
        self.root_dataset_folder = root_dataset_folder
        self.folder_paths = [os.path.join(root_dataset_folder, folder) for folder in os.listdir(root_dataset_folder) if os.path.isdir(os.path.join(root_dataset_folder, folder))]
        self.validator = validator
        self.feature_extractor = feature_extractor
        self.data_list = []

    def process_audio(self, folder_path):
        language = folder_path.name
        audio_files = glob.glob(os.path.join(folder_path, '*.mp3'))

        batch_size = 1000

        for i in range(0, len(audio_files), batch_size):
            batch_files = audio_files[i:i + batch_size]

            for audio_file in batch_files:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if self.validator.validate(audio_file):
                            audio, sample_rate = librosa.load(audio_file, sr=None)
                            if self.validator.validate_audio_content(audio):
                                mfccs = self.feature_extractor.extract_mfccs(audio, sample_rate)
                                self.data_list.append({'File': audio_file, 'Language': language, 'MFCCs': mfccs})
                except Exception as e:
                    print(f"Invalid audio file: {audio_file}. Error: {e}")

    def process_all_folders(self):

        invalid_count_before = self.validator.get_invalid_count()

        for folder_path in Path(self.root_dataset_folder).iterdir():
            if folder_path.is_dir():
                self.process_audio(folder_path)

        # Access invalid count after processing folders
        invalid_count_after = self.validator.get_invalid_count()

        total_invalid_count = invalid_count_after - invalid_count_before
        print(f"Total invalid audio files: {total_invalid_count}")
        print(f"Total items in data_list: {len(self.data_list)}")

    def save_to_csv(self, csv_file_path):
        print("Data List:", self.data_list)
        df = pd.DataFrame(self.data_list)
        df.to_csv(csv_file_path, index=False)
        print(f"CSV file saved to: {csv_file_path}")



root_folder = '/Users/vanshika/PycharmProjects/Language Classification/Language Detection Dataset'
csv_file_path = '/Users/vanshika/PycharmProjects/Language Classification/processed_data.csv'

validator = AudioValidator()
feature_extractor = FeatureExtractor()

audio_processor = AudioProcessor(root_folder, validator, feature_extractor)
audio_processor.process_all_folders()
audio_processor.save_to_csv(csv_file_path)
