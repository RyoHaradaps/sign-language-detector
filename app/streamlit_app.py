import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
import mediapipe as mp
from tensorflow.keras.models import model_from_json
from gtts import gTTS
import pyttsx3
import threading
from collections import Counter

# --- Page Configuration ---
st.set_page_config(
    page_title="ISL Translator",
    page_icon="🤟",
    layout="wide"
)

# --- Constants ---
SEQ_LEN = 60
FEATURE_DIM = 126
CONFIDENCE_THRESHOLD = 0.7

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        with open('models/model_architecture.json', 'r') as f:
            model_json = f.read()
        model = model_from_json(model_json)
        model.load_weights('models/model_weights.h5')
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_classes():
    try:
        with open('data/processed/classes.txt', 'r') as f:
            return [line.strip() for line in f.readlines()]
    except:
        return ['namaste', 'thank_you', 'please', 'sorry']

model = load_model()
classes = load_classes()

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# --- Helper Functions ---
def extract_landmarks(frame):
    """Extract hand landmarks from frame"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    data = []
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            for lm in hand.landmark:
                data.extend([lm.x, lm.y, lm.z])
    while len(data) < FEATURE_DIM:
        data.append(0.0)
    return np.array(data[:FEATURE_DIM]), results

def process_video(video_path):
    """Process uploaded video and return predictions"""
    cap = cv2.VideoCapture(video_path)
    buffer = []
    predictions = []
    confidence_scores = []
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        landmarks, _ = extract_landmarks(frame)
        buffer.append(landmarks)
        
        if len(buffer) > SEQ_LEN:
            buffer.pop(0)
        
        if len(buffer) == SEQ_LEN:
            input_seq = np.array(buffer).reshape(1, SEQ_LEN, FEATURE_DIM)
            preds = model.predict(input_seq, verbose=0)[0]
            idx = np.argmax(preds)
            conf = preds[idx]
            
            if conf > CONFIDENCE_THRESHOLD:
                predictions.append(classes[idx])
                confidence_scores.append(conf)
            else:
                predictions.append("???")
                confidence_scores.append(conf)
        
        # Process every 15th frame for speed
        if frame_count % 15 != 0:
            continue
    
    cap.release()
    
    if predictions:
        most_common = Counter(predictions).most_common(1)
        if most_common[0][0] != "???":
            return most_common[0][0], max(confidence_scores) if confidence_scores else 0.0
        else:
            return "No clear sign detected", 0.0
    return "No frames processed", 0.0

# --- Streamlit UI ---
st.title("🤟 Indian Sign Language Translator")
st.markdown("Upload a video or use your webcam to translate ISL to English text.")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    **Supported Signs:**
    - Namaste
    - Thank You
    - Please
    - Sorry
    
    **How It Works:**
    1. MediaPipe extracts hand landmarks
    2. Landmarks are buffered into sequences
    3. LSTM model predicts the sign
    4. Text is displayed with confidence
    """)
    
    st.header("Status")
    if model is not None:
        st.success(f"✅ Model loaded ({len(classes)} signs)")
    else:
        st.error("❌ Model not loaded")

# Tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload Video", "🎥 Real-Time Webcam", "📊 About"])

# --- Tab 1: Upload Video ---
with tab1:
    st.header("Upload a Pre-Recorded Video")
    uploaded_file = st.file_uploader(
        "Choose a video file...", 
        type=['mp4', 'avi', 'mov', 'mkv', 'webm']
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        st.video(tmp_path)
        
        if st.button("🔍 Translate Video"):
            with st.spinner("Processing video..."):
                prediction, confidence = process_video(tmp_path)
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Prediction:** {prediction}")
            with col2:
                st.info(f"Confidence: {confidence:.2f}")
            
            # Text-to-Speech
            if st.button("🔊 Read Aloud"):
                if prediction and prediction not in ["No clear sign detected", "No frames processed"]:
                    try:
                        engine = pyttsx3.init()
                        engine.say(prediction)
                        engine.runAndWait()
                    except Exception as e:
                        st.error(f"Text-to-speech error: {e}")
        
        # Clean up
        os.unlink(tmp_path)

# --- Tab 2: Real-Time Webcam ---
with tab2:
    st.header("Real-Time Webcam Detection")
    st.warning("⚠️ Uses your phone as webcam via IP Webcam (192.168.1.3:8080)")
    
    # Phone IP configuration
    phone_ip_input = st.text_input("Phone IP Address", value="192.168.1.3:8080")
    use_camera = st.button("🎥 Start Webcam Detection")
    
    if use_camera:
        if model is None:
            st.error("Model not loaded. Please check model files.")
        else:
            try:
                # Construct the video URL (use the user input)
                video_url = f"http://{phone_ip_input}/video"
                st.info(f"Connecting to: {video_url}")
                
                cap = cv2.VideoCapture(video_url)
                if not cap.isOpened():
                    st.error("Cannot connect to phone camera. Make sure IP Webcam is running.")
                    st.stop()
                
                st.info("Camera connected! Show your signs...")
                
                buffer = []
                prediction = "Waiting..."
                confidence = 0.0
                
                # Create placeholders
                frame_placeholder = st.empty()
                text_placeholder = st.empty()
                
                # Run detection loop
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Camera disconnected. Please check connection.")
                        break
                    
                    # Flip frame for mirror effect
                    frame = cv2.flip(frame, 1)
                    
                    # Extract landmarks
                    landmarks, _ = extract_landmarks(frame)
                    buffer.append(landmarks)
                    
                    if len(buffer) > SEQ_LEN:
                        buffer.pop(0)
                    
                    if len(buffer) == SEQ_LEN:
                        input_seq = np.array(buffer).reshape(1, SEQ_LEN, FEATURE_DIM)
                        preds = model.predict(input_seq, verbose=0)[0]
                        idx = np.argmax(preds)
                        conf = preds[idx]
                        
                        if conf > CONFIDENCE_THRESHOLD:
                            prediction = classes[idx]
                            confidence = conf
                        else:
                            prediction = "???"
                            confidence = conf
                    
                    # Display frame with overlay
                    cv2.putText(frame, f"Sign: {prediction}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # Convert BGR to RGB for Streamlit
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Update prediction text
                    text_placeholder.markdown(f"### Prediction: **{prediction}** (Confidence: {confidence:.2f})")
                
                cap.release()
                cv2.destroyAllWindows()
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
                
# --- Tab 3: About ---
with tab3:
    st.header("About This Project")
    st.markdown("""
    ### 🎯 Project Overview
    This is a real-time Indian Sign Language (ISL) translator built with:
    - **MediaPipe**: For hand landmark extraction
    - **LSTM Model**: For sequence prediction
    - **Streamlit**: For web interface
    - **IP Webcam**: For phone camera integration
    
    ### 🔬 Model Performance
    - **Accuracy**: 91.67% on test set
    - **Architecture**: Bidirectional LSTM with Dropout
    - **Training**: 4 signs, 30+ samples each
    
    ### 📱 How to Use
    1. **Upload Video**: Translate pre-recorded ISL videos
    2. **Real-Time Webcam**: Use your phone camera for live translation
    3. **Read Aloud**: Listen to the translated text
    
    ### 🚀 Future Improvements
    - More signs (100+)
    - Sentence-level translation
    - Multi-language support
    - Mobile app deployment
    """)

st.markdown("---")
st.markdown("🤟 **Made with ❤️ for the Deaf Community**")