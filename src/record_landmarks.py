import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create directories
os.makedirs('data/raw_landmarks', exist_ok=True)

# =====================================================
# INDIAN SIGN LANGUAGE (ISL) GESTURES
# Based on ISLRTC official standards
# =====================================================
SIGNS = {
    0: "namaste",
    1: "thank_you",
    2: "help",
    3: "yes",
    4: "no",
    5: "please",
    6: "sorry",
    7: "water",
    8: "food",
    9: "stop"
}

class SignLanguageRecorder:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def extract_landmarks(self, frame):
        """Extract hand landmarks from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        landmarks_data = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmarks_data.extend([lm.x, lm.y, lm.z])
        
        # Pad to 126 features (2 hands × 21 landmarks × 3)
        while len(landmarks_data) < 126:
            landmarks_data.append(0.0)
        landmarks_data = landmarks_data[:126]
        
        return np.array(landmarks_data), results
    
    def record_sequence(self, sign_name, duration_seconds=2):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        sequence = []
        
        # Countdown
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            if ret:
                # Display sign name and countdown
                cv2.putText(frame, f"Sign: {sign_name}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, f"Get ready... {i}", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Recording', frame)
                cv2.waitKey(1000)
        
        # Recording
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration_seconds:
            ret, frame = cap.read()
            if not ret:
                continue
                
            landmarks, _ = self.extract_landmarks(frame)
            sequence.append(landmarks)
            frame_count += 1
            
            cv2.putText(frame, f"Recording: {sign_name} - Frame {frame_count}", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Time: {time.time() - start_time:.1f}/{duration_seconds}s", 
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Recording', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return np.array(sequence)
    
    def save_sequence(self, sequence, sign_name, sample_id):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/raw_landmarks/{sign_name}_{sample_id}_{timestamp}.csv"
        df = pd.DataFrame(sequence)
        df['label'] = sign_name
        df.to_csv(filename, index=False)
        print(f"✓ Saved: {filename}")
        return filename

def main():
    recorder = SignLanguageRecorder()
    
    print("\n" + "="*60)
    print("    INDIAN SIGN LANGUAGE (ISL) DATA RECORDER")
    print("="*60)
    print("\nAvailable ISL gestures:")
    for idx, sign in SIGNS.items():
        print(f"  {idx}: {sign}")
    
    print("\n📖 Learn the correct gestures from ISLRTC:")
    print("   https://www.youtube.com/@ISLRTC")
    print("\nInstructions:")
    print("  - Enter sign number (0-9) to record that gesture")
    print("  - Perform the gesture clearly for 2 seconds")
    print("  - Record 30-50 samples per sign for best results")
    print("  - Vary speed, angle, and lighting")
    print("  - Type 'list' to see signs again")
    print("  - Type 'quit' to exit")
    
    sample_counts = {sign: 0 for sign in SIGNS.values()}
    
    while True:
        print("\n" + "-"*60)
        user_input = input("Enter sign number to record (or 'quit'): ").strip().lower()
        
        if user_input == 'quit':
            break
        elif user_input == 'list':
            print("\nCurrent progress:")
            for idx, sign in SIGNS.items():
                print(f"  {idx}: {sign} → {sample_counts[sign]} samples")
            continue
        
        try:
            sign_idx = int(user_input)
            if sign_idx not in SIGNS:
                print(f"Invalid. Choose 0-{len(SIGNS)-1}")
                continue
                
            sign_name = SIGNS[sign_idx]
            sample_counts[sign_name] += 1
            
            print(f"\n▶ Recording: {sign_name} (Sample #{sample_counts[sign_name]})")
            print("Look at the camera window and perform the sign after countdown.")
            
            sequence = recorder.record_sequence(sign_name, duration_seconds=2)
            
            if len(sequence) > 0:
                recorder.save_sequence(sequence, sign_name, sample_counts[sign_name])
                print(f"✅ Recorded {len(sequence)} frames for '{sign_name}'")
            else:
                print("❌ No frames recorded. Check camera and try again.")
                sample_counts[sign_name] -= 1
                
        except ValueError:
            print("Invalid input. Enter a number, 'list', or 'quit'")
    
    print("\n" + "="*60)
    print("RECORDING SESSION SUMMARY")
    print("="*60)
    for sign, count in sample_counts.items():
        print(f"  {sign}: {count} samples")
    print("\nKeep recording until you have 30+ per sign!")
    print("="*60)

if __name__ == "__main__":
    main()