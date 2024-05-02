'''
import uvicorn
from fastapi import FastAPI
from keras.models import load_model
import numpy as np
from keras.preprocessing import sequence
import pickle
from MfccInputs import MFCCInput

app = FastAPI()
# pkl_inn = open('/Users/vanshika/PycharmProjects/Language Classification/language_classification_model.h5', 'rb')
classifier = load_model('/Users/vanshika/PycharmProjects/Language Classification/language_classification_model.h5')
pickle_in = open('label_encoder.pkl', 'rb')
labelencoder = pickle.load(pickle_in)



# Index route, opens automatically on https:
@app.get('/')
def index():
    return {'message': 'Hello.'}


# function to predict language.
async def predict_language(mfcc_input: MFCCInput):
    mfcc = np.array(mfcc_input.MFCCs)
    mfcc_padded = sequence.pad_sequences([mfcc], dtype='float32', padding='post', truncating='post')
    mfcc_padded = mfcc_padded.reshape((mfcc_padded.shape[0], mfcc_padded.shape[1], 1))

    # Make predictions using the loaded Keras model
    predictions = classifier.predict(mfcc_padded)
    
    # Decode predictions
    predicted_language_indices = np.argmax(predictions, axis=1)
    predicted_languages = [labelencoder.classes_[index] for index in predicted_language_indices]
    return predicted_languages


# Prediction funtionality, make prediction from the passed JSON data
# and return the predicted language.
@app.post("/predict")

async def predict(mfcc_input: MFCCInput):
    return await predict_language(mfcc_input)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

# uvicorn app:app --reload
'''

import uvicorn
from fastapi import FastAPI
from keras.models import load_model
import numpy as np
from keras.preprocessing import sequence
import pickle
from MfccInputs import MFCCInput

app = FastAPI()

# Load the Keras model
classifier = load_model('/Users/vanshika/PycharmProjects/Language Classification/language_classification_model.h5')

# Load the label encoder
with open('label_encoder.pkl', 'rb') as pickle_in:
    labelencoder = pickle.load(pickle_in)

# Index route
@app.get('/')
def index():
    return {'message': 'Hello.'}

# Function to predict language
async def predict_language(mfcc_input: MFCCInput):
    mfcc = np.array(mfcc_input.MFCCs)
    mfcc_padded = sequence.pad_sequences([mfcc], dtype='float32', padding='post', truncating='post')
    mfcc_padded = mfcc_padded.reshape((mfcc_padded.shape[0], mfcc_padded.shape[1], 1))

    # Make predictions using the loaded Keras model
    predictions = classifier.predict(mfcc_padded)
    
    # Decode predictions
    predicted_language_indices = np.argmax(predictions, axis=1)
    predicted_languages = [labelencoder.classes_[index] for index in predicted_language_indices]
    return predicted_languages

# Prediction functionality
@app.post("/predict")
async def predict(mfcc_input: MFCCInput):
    return await predict_language(mfcc_input)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
