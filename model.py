import PyPDF2
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
import json
import matplotlib.pyplot as plt # type: ignore

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file_path):
    with open(pdf_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

# Extract text from the provided PDF
pdf_text = extract_text_from_pdf('Ryan.pdf')

# Preprocess the extracted text
corpus = pdf_text.split('\n')
corpus = [line.lower().replace('\n', ' ').strip() for line in corpus if line.strip()]

# Tokenize and preprocess the corpus
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# Save the tokenizer for later use in prediction
with open('tokenizer.json', 'w') as f:
    json.dump(tokenizer.to_json(), f)

# Prepare input sequences for training
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Adjust max_sequence_length based on the length of the longest sequence
max_sequence_length = max([len(x) for x in input_sequences])

# Pad sequences for consistent input shape
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Prepare the input (X) and output (y) data
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

# Define the LSTM model with added Dropout for regularization
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_length - 1))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(150, return_sequences=False))
model.add(Dense(total_words, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))

# Add checkpoints and early stopping
checkpoint = ModelCheckpoint('best_model.keras', monitor='loss', save_best_only=True, verbose=1)

early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1)

# Train the model with callbacks
history = model.fit(X, y, epochs=50, batch_size=64, verbose=1, callbacks=[checkpoint, early_stopping])

# Save the final model
model.save('final_lstm_model.h5')
print("Model trained and saved as 'final_lstm_model.h5'.")

# Plot the training loss
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Over Epochs')
plt.show()
