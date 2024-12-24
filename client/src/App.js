import React, { useState } from 'react';
import axios from 'axios';
import './App.css'; // External CSS for styling

const App = () => {
  const [inputText, setInputText] = useState('');
  const [predictedWord, setPredictedWord] = useState('');

  const handlePredict = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', {
        input_text: inputText,
      });
      setPredictedWord(response.data.predicted_word);
    } catch (error) {
      console.error('Error fetching prediction:', error);
    }
  };

  return (
    <div className="app-container">
      <div className="content">
        <h1 className="title">Next Word Prediction</h1>
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Enter text"
          className="input-text"
        />
        <button onClick={handlePredict} className="predict-button">
          Predict
        </button>
        {predictedWord && <p className="result">Predicted Next Word: {predictedWord}</p>}
      </div>
    </div>
  );
};

export default App;
