import numpy as np
import librosa
from joblib import load
from tensorflow.keras.models import load_model

scaler = load('scaler.pkl')
encoder = load('label_encoder.pkl')
model = load_model('cnn_model_2.1_.h5')

def emotion_classifier(file_path):
    # Load and preprocess the audio file
    data, sr = librosa.load(file_path, duration=2.5, offset=0.6)
    features = np.array([librosa.feature.mfcc(y=data, sr=sr)])

    # Scale the features
    X_scaled = scaler.transform(features)

    # Reshape the features for the model
    X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1, 1))

    # Predict the emotion
    predictions = model.predict(X_scaled)
    predicted_emotion = encoder.inverse_transform(np.argmax(predictions, axis=1))

    return predicted_emotion[0], predictions[0]

# Usage example:
predicted_emotion, predictions = emotion_classifier('path_to_your_audio_file.wav')
print(f'Predicted Emotion: {predicted_emotion}')
print(f'Prediction Probabilities: {predictions}')
