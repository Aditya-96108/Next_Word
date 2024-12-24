# Next Word Prediction using LSTM

This project uses a Long Short-Term Memory (LSTM) model to predict the next word in a sentence based on a given input. It is built using TensorFlow and Keras for model training, and React for the front-end to interact with the model.

## Overview

The project consists of two parts:
1. **Backend**: A Python-based LSTM model to predict the next word based on input text.
2. **Frontend**: A React application that allows users to input text and receive predictions from the trained LSTM model.

### Features:
- Predicts the next word based on a given sentence.
- Built using a large dataset extracted from a PDF (Ryan.pdf).
- Uses LSTM for training the model.

## Setup

### Backend (Python)
1. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
To train the model:

bash
Copy code
python model.py
The trained model (final_lstm_model.h5) and tokenizer (tokenizer.json) will be saved in the project folder.

Start the Flask server to provide predictions:

bash
Copy code
python app.py
The backend will be available at http://127.0.0.1:5000.

Frontend (React)
Navigate to the client directory:

bash
Copy code
cd client
Install dependencies:

bash
Copy code
npm install
Run the React application:

bash
Copy code
npm start
Open your browser and visit http://localhost:3000 to interact with the prediction model.
