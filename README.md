# Human Scream Detection System - Code Documentation

## Core Modules Implementation

### 1. Audio Capture Module
```python
import pyaudio
import numpy as np
import wave

class AudioCapture:
    def __init__(self, sample_rate=44100, chunk_size=1024, channels=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.p = pyaudio.PyAudio()
        
    def start_stream(self):
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
    def read_audio(self):
        data = self.stream.read(self.chunk_size)
        return np.frombuffer(data, dtype=np.float32)
        
    def stop_stream(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
```

### 2. Audio Preprocessing Module
```python
import librosa
import numpy as np

class AudioPreprocessor:
    def __init__(self, sample_rate=44100, n_mfcc=13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        
    def reduce_noise(self, audio_data):
        # Simple noise reduction using median filtering
        return scipy.signal.medfilt(audio_data)
        
    def extract_features(self, audio_data):
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio_data,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc
        )
        return mfccs
        
    def normalize_features(self, features):
        return (features - np.mean(features)) / np.std(features)
```

### 3. ML Model Implementation
```python
import tensorflow as tf
from tensorflow.keras import layers, models

class ScreamDetectionModel:
    def __init__(self):
        self.model = self._build_model()
        
    def _build_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(13, 44, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
        
    def train(self, X_train, y_train, epochs=10, batch_size=32):
        return self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )
        
    def predict(self, X):
        return self.model.predict(X)
```

### 4. Alert System Implementation
```python
import smtplib
from email.mime.text import MIMEText
import json

class AlertSystem:
    def __init__(self, config_path='config/alert_config.json'):
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
            
    def generate_alert(self, confidence_score, location):
        if confidence_score > self.config['threshold']:
            alert_data = {
                'timestamp': datetime.now().isoformat(),
                'confidence': confidence_score,
                'location': location
            }
            self._send_alert(alert_data)
            
    def _send_alert(self, alert_data):
        msg = MIMEText(json.dumps(alert_data, indent=2))
        msg['Subject'] = 'ALERT: Scream Detected'
        msg['From'] = self.config['email']['sender']
        msg['To'] = self.config['email']['recipient']
        
        with smtplib.SMTP(self.config['email']['smtp_server']) as server:
            server.login(
                self.config['email']['username'],
                self.config['email']['password']
            )
            server.send_message(msg)
```

### 5. Main Application
```python
class ScreamDetectionSystem:
    def __init__(self):
        self.audio_capture = AudioCapture()
        self.preprocessor = AudioPreprocessor()
        self.model = ScreamDetectionModel()
        self.alert_system = AlertSystem()
        
    def run(self):
        self.audio_capture.start_stream()
        try:
            while True:
                # Capture audio
                audio_data = self.audio_capture.read_audio()
                
                # Preprocess
                cleaned_audio = self.preprocessor.reduce_noise(audio_data)
                features = self.preprocessor.extract_features(cleaned_audio)
                normalized_features = self.preprocessor.normalize_features(features)
                
                # Predict
                prediction = self.model.predict(normalized_features)
                
                # Handle alerts
                if prediction[0][1] > 0.75:  # Scream probability threshold
                    self.alert_system.generate_alert(
                        confidence_score=prediction[0][1],
                        location='Default Location'
                    )
                    
        except KeyboardInterrupt:
            self.audio_capture.stop_stream()
```

### 6. Configuration (config/config.yaml)
```yaml
audio:
  sample_rate: 44100
  chunk_size: 1024
  channels: 1
  
model:
  n_mfcc: 13
  threshold: 0.75
  
alert:
  email:
    smtp_server: "smtp.gmail.com"
    port: 587
    sender: "sender@example.com"
    recipient: "alert@example.com"
```

### 7. Usage Example
```python
if __name__ == "__main__":
    # Initialize and run the system
    system = ScreamDetectionSystem()
    system.run()
```

### 8. Testing Example
```python
import unittest

class TestScreamDetection(unittest.TestCase):
    def setUp(self):
        self.model = ScreamDetectionModel()
        self.preprocessor = AudioPreprocessor()
        
    def test_feature_extraction(self):
        # Create dummy audio data
        audio_data = np.random.rand(44100)
        features = self.preprocessor.extract_features(audio_data)
        self.assertEqual(features.shape[0], 13)  # Expected MFCC features
        
    def test_model_prediction(self):
        # Create dummy input
        X = np.random.rand(1, 13, 44, 1)
        prediction = self.model.predict(X)
        self.assertEqual(prediction.shape, (1, 2))  # Binary classification
```

## Requirements
```txt
numpy==1.21.0
tensorflow==2.7.0
librosa==0.8.1
pyaudio==0.2.11
scipy==1.7.0
pyyaml==5.4.1
```

