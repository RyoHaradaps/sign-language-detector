# 👐 Real-time Dynamic Sign Language Detector

A real-time sign language recognition system that detects dynamic gestures using **MediaPipe** for landmark extraction and **LSTM with Attention** for sequence classification.

## 🎯 Novel Approach
- **Velocity + relative position features** for person-invariant detection
- **Bidirectional LSTM + Temporal Attention** for motion pattern recognition
- **Real-time continuous detection** without button triggers

## 🛠️ Tech Stack
- Python 3.9
- MediaPipe (hand landmark extraction)
- TensorFlow 2.15 (LSTM + Attention)
- OpenCV (real-time video processing)
- WSL2 + CUDA (GPU acceleration)

## 📁 Project Structure
sign_language_detector/
├── data/
│ ├── raw_landmarks/ # Recorded landmark sequences (CSV)
│ └── processed/ # Preprocessed data for training
├── src/
│ ├── record_landmarks.py # Data collection script
│ ├── preprocess.py # Data preprocessing
│ ├── train_model.py # LSTM model training
│ └── realtime_detector.py # Live inference
├── models/ # Saved trained models
├── notebooks/ # EDA and experimentation
└── demo/ # Demo video & screenshots


## 🚀 Current Status
- ✅ Environment setup (WSL2 + GPU)
- ✅ MediaPipe hand tracking working
- ✅ Data recording script complete
- 🔄 Data collection in progress (30+ samples per sign)
- ⏳ Model training (upcoming)
- ⏳ Real-time deployment (upcoming)

## 📊 Signs Supported
- hello, thank_you, help, yes, no
- *(expandable to more signs)*

## 🏗️ Model Architecture
Input (30 frames × 126 landmarks)
→ Bidirectional LSTM (128 units)
→ Attention Layer
→ Dropout (0.3)
→ Bidirectional LSTM (64 units)
→ Dense (32, ReLU)
→ Dense (num_classes, Softmax)


## 🔧 Installation
```bash
conda create -n signlang python=3.9
conda activate signlang
pip install opencv-python mediapipe tensorflow numpy pandas

📈 Results
(to be added after training)
