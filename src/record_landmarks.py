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

# Sign labels (you can add more)
SIGNS = {
    0: "hello",
    1: "thank_you", 
    2: "help",
    3: "yes",
    4: "no",
    5: "please",
    6: "sorry",
    7: "i_love_you"
}

class SignLanguageRecorder:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.recording = False
        self.current_sign = None
        self.current_sequence = []
        self.sequence_count = 0
        
    def extract_landmarks(self, frame):
        """Extract hand landmarks from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        landmarks_data = []
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks on frame (optional)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract 21 landmarks × 3 coordinates (x, y, z)
                for lm in hand_landmarks.landmark:
                    landmarks_data.extend([lm.x, lm.y, lm.z])
                
                # Pad if only one hand detected (max 2 hands, 21 landmarks each, 3 coords)
                # We'll standardize to 2 hands (126 features: 2*21*3)
                
        # Pad to 2 hands (126 features)
        while len(landmarks_data) < 126:
            landmarks_data.append(0.0)
        # Truncate if more than 126
        landmarks_data = landmarks_data[:126]
        
        return np.array(landmarks_data), results
    
    def record_sequence(self, sign_name, duration_seconds=2, fps=30):
        """Record a sequence of frames for a specific sign"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        sequence = []
        
        # Countdown
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            if ret:
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
            
            # Show recording indicator
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
        """Save recorded sequence to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/raw_landmarks/{sign_name}_{sample_id}_{timestamp}.csv"
        
        # Convert to DataFrame
        df = pd.DataFrame(sequence)
        df['label'] = sign_name
        df.to_csv(filename, index=False)
        print(f"Saved: {filename}")
        return filename

def main():
    recorder = SignLanguageRecorder()
    
    print("\n" + "="*50)
    print("SIGN LANGUAGE DATA RECORDER")
    print("="*50)
    print("\nAvailable signs:")
    for idx, sign in SIGNS.items():
        print(f"  {idx}: {sign}")
    
    print("\nInstructions:")
    print("  - Enter sign number (0-7) to record that sign")
    print("  - Enter 'list' to show signs again")
    print("  - Enter 'quit' to exit")
    print("  - Each sign: 2-second recording, 30+ frames")
    print("  - Record 30-50 samples per sign for good results")
    
    # Track samples per sign
    sample_counts = {sign: 0 for sign in SIGNS.values()}
    
    while True:
        print("\n" + "-"*50)
        user_input = input("Enter sign number to record (or 'quit'): ").strip().lower()
        
        if user_input == 'quit':
            break
        elif user_input == 'list':
            for idx, sign in SIGNS.items():
                print(f"  {idx}: {sign} (recorded: {sample_counts[sign]})")
            continue
        
        try:
            sign_idx = int(user_input)
            if sign_idx not in SIGNS:
                print(f"Invalid sign number. Choose 0-{len(SIGNS)-1}")
                continue
                
            sign_name = SIGNS[sign_idx]
            sample_counts[sign_name] += 1
            
            print(f"\nRecording: {sign_name} (Sample #{sample_counts[sign_name]})")
            print("Look at the camera window...")
            
            # Record sequence (2 seconds)
            sequence = recorder.record_sequence(sign_name, duration_seconds=2)
            
            if len(sequence) > 0:
                recorder.save_sequence(sequence, sign_name, sample_counts[sign_name])
                print(f"✓ Recorded {len(sequence)} frames for '{sign_name}'")
            else:
                print("✗ No frames recorded. Check camera.")
                sample_counts[sign_name] -= 1
                
        except ValueError:
            print("Invalid input. Enter a number, 'list', or 'quit'")
    
    print("\n" + "="*50)
    print("Recording session summary:")
    for sign, count in sample_counts.items():
        print(f"  {sign}: {count} samples")
    print("="*50)

if __name__ == "__main__":
    main()