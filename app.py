from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
from flask_cors import CORS
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load the trained model
lstm_model = load_model('lstm_model.h5')

# Load the tokenizer (matching the tokenizer used during training)
with open('tokenizer.json', 'r') as f:
    tokenizer = tokenizer_from_json(json.load(f))

# Create word-to-index and index-to-word mappings based on the tokenizer
word_to_int = tokenizer.word_index
int_to_word = {i: word for word, i in word_to_int.items()}

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Predict next word
@app.route('/predict', methods=['POST'])
def predict_next_word():
    data = request.get_json()
    input_text = data['input_text']
    
    # Tokenize input text
    input_words = input_text.split()
    input_seq = [word_to_int.get(word, 0) for word in input_words]  # Get index for each word
    
    # Handle missing words (index 0 means unknown word)
    if len(input_seq) == 0:
        return jsonify({'predicted_word': 'unknown'})
    
    input_seq = pad_sequences([input_seq], maxlen=10, padding='pre')
    
    # Predict the next word
    predicted_probs = lstm_model.predict(input_seq)
    predicted_word_idx = np.argmax(predicted_probs)
    
    # Get the predicted word
    predicted_word = int_to_word.get(predicted_word_idx, 'unknown')  # Default to 'unknown' if index is not found

    return jsonify({'predicted_word': predicted_word})

if __name__ == '__main__':
    app.run(debug=True)
