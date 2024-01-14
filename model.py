import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
import tensorflow.keras.optimizers as optimizers
from keras.preprocessing import sequence
import ast
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="absl")


data = pd.read_csv('processed_data.csv')

label_encoder = LabelEncoder()
data['Language_encoded'] = label_encoder.fit_transform(data['Language'])
data['MFCCs'] = data['MFCCs'].apply(ast.literal_eval)

max_frames = max(len(mfcc) for mfcc in data['MFCCs'])

# Pad all MFCCs to have the same number of frames
X = np.array(data['MFCCs'])
print("Shape of X before padding:", X.shape)

# Pad sequences to have the same length
X_padded = sequence.pad_sequences(X, dtype='float32', padding='post', truncating='post')
X_padded = X_padded.reshape((X_padded.shape[0], X_padded.shape[1], 1))

print("Shape of X after padding:", X_padded.shape)

y = data['Language_encoded']
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Convert labels to categorical one-hot encoding
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# Define the model
model = Sequential()
model.add(Dense(256, input_shape=(X_padded.shape[1],), activation='relu'))  # Changed input shape
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_encoded, epochs=20, batch_size=32, validation_data=(X_test, y_test_encoded))
warnings.resetwarnings()

# Evaluate the model on the test set
accuracy = model.evaluate(X_test, y_test_encoded)[1]
print(f"Accuracy on the test set: {accuracy}")

model.save('language_classification_model.h5')
print("model saved!!")