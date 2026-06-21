import os
import sys
import time
from datetime import datetime
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create directories
os.makedirs('data/raw_landmarks', exist_ok=True)

# Signs based on iSign frequency
SIGNS = {
    0: "namaste",
    1: "thank_you",
    2: "please",
    3: "sorry",
    4: "help",
    5: "yes",
    6: "no",
    7: "stop",
    8: "water",
    9: "food",
    10: "good",
    11: "bad",
    12: "love",
    13: "friend",
    14: "work",
    15: "school"
}

class IsignRecorder:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.target_samples = 50
        self.phone_ip = "http://192.0.0.4:8080/video"  # Update to your IP
    
    def extract_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        data = []
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                for lm in hand.landmark:
                    data.extend([lm.x, lm.y, lm.z])
        while len(data) < 126:
            data.append(0.0)
        return np.array(data[:126]), results
    
    def count_existing_samples(self, sign_name):
        """Count how many samples already exist for a sign"""
        count = 0
        data_dir = 'data/raw_landmarks'
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.startswith(f"{sign_name}_") and file.endswith('.csv'):
                    count += 1
        return count
    
    def record_sequence(self, sign_name, duration=3):
        print(f"📱 Connecting to phone: {self.phone_ip}")
        cap = cv2.VideoCapture(self.phone_ip)
        
        if not cap.isOpened():
            print("❌ Cannot connect to phone camera. Make sure IP Webcam is running.")
            return np.array([])
        
        print("✅ Connected! Recording will start in 3 seconds...")
        
        # Warm up and countdown (no GUI)
        for i in range(3, 0, -1):
            print(f"   Starting in {i}...")
            ret, _ = cap.read()
            if not ret:
                print("⚠️ Camera not responding")
                time.sleep(0.5)
            time.sleep(1)
        
        print("🎬 Recording now...")
        sequence = []
        start = time.time()
        frame_count = 0
        
        while time.time() - start < duration:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            # Resize for faster processing
            frame = cv2.resize(frame, (640, 480))
            
            lm, _ = self.extract_landmarks(frame)
            sequence.append(lm)
            
            # Progress indicator
            if frame_count % 5 == 0:
                print(f"   Frame {len(sequence)}", end="\r")
        
        cap.release()
        print(f"\n✅ Recorded {len(sequence)} frames")
        return np.array(sequence)
    
    def save(self, seq, sign, sample_id):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join('data/raw_landmarks', f"{sign}_{sample_id}_{ts}.csv")
        df = pd.DataFrame(seq)
        df['label'] = sign
        df.to_csv(path, index=False)
        print(f"💾 Saved: {path} ({len(seq)} frames)")

def main():
    recorder = IsignRecorder()
    
    print("\n" + "="*60)
    print("   iSign Dataset Recorder (Headless Mode)")
    print("="*60)
    print(f"\n📱 Phone IP: {recorder.phone_ip}")
    print("   Make sure IP Webcam is running on your phone!")
    
    # Count existing samples for each sign
    print("\n📊 Current Progress:")
    for idx, sign in SIGNS.items():
        existing = recorder.count_existing_samples(sign)
        print(f"  {idx}: {sign} → {existing}/{recorder.target_samples} existing")
    
    print("\nAvailable signs to record:")
    for idx, sign in SIGNS.items():
        print(f"  {idx}: {sign}")
    
    print(f"\n🎯 Target: {recorder.target_samples} samples per sign")
    
    print("\nCommands:")
    print("  - Enter sign number (0-15) to record that sign")
    print("  - Type 'auto' to record all signs from start")
    print("  - Type 'auto X' to start from sign X (e.g., 'auto 1' for thank_you)")
    print("  - Type 'list' to show progress")
    print("  - Type 'quit' to exit")
    
    counts = {sign: recorder.count_existing_samples(sign) for sign in SIGNS.values()}
    
    while True:
        print("\n" + "-"*50)
        user_input = input("Enter command: ").strip()
        
        if user_input == 'quit':
            break
        elif user_input == 'list':
            print("\n📊 Progress:")
            for idx, sign in SIGNS.items():
                print(f"  {idx}: {sign} → {counts[sign]}/{recorder.target_samples}")
            continue
        elif user_input.startswith('auto'):
            # Parse start index
            parts = user_input.split()
            start_idx = 0
            if len(parts) > 1:
                try:
                    start_idx = int(parts[1])
                    if start_idx not in SIGNS:
                        print(f"❌ Invalid sign number. Choose 0-{len(SIGNS)-1}")
                        continue
                except ValueError:
                    print("❌ Invalid number. Usage: 'auto X' where X is 0-15")
                    continue
            
            print(f"\n🔄 Starting auto-recording from sign {start_idx}: {SIGNS[start_idx]}")
            
            for idx in range(start_idx, len(SIGNS)):
                sign = SIGNS[idx]
                print(f"\n{'='*50}")
                print(f"📌 Now recording: {sign} (Index {idx})")
                print(f"{'='*50}")
                
                while counts[sign] < recorder.target_samples:
                    counts[sign] += 1
                    print(f"\n🎬 Recording: {sign} (#{counts[sign]}/{recorder.target_samples})")
                    seq = recorder.record_sequence(sign, duration=3)
                    if len(seq) > 5:
                        recorder.save(seq, sign, counts[sign])
                    else:
                        counts[sign] -= 1
                        print("❌ Failed (not enough frames), retrying...")
                
                print(f"✅ Completed {sign} with {counts[sign]} samples!")
            
            print(f"\n🎉 All signs recorded successfully!")
            break
        
        elif user_input == 'auto':
            # Default: start from 0
            start_idx = 0
            print(f"\n🔄 Starting auto-recording from sign {start_idx}: {SIGNS[start_idx]}")
            
            for idx in range(start_idx, len(SIGNS)):
                sign = SIGNS[idx]
                print(f"\n{'='*50}")
                print(f"📌 Now recording: {sign} (Index {idx})")
                print(f"{'='*50}")
                
                while counts[sign] < recorder.target_samples:
                    counts[sign] += 1
                    print(f"\n🎬 Recording: {sign} (#{counts[sign]}/{recorder.target_samples})")
                    seq = recorder.record_sequence(sign, duration=3)
                    if len(seq) > 5:
                        recorder.save(seq, sign, counts[sign])
                    else:
                        counts[sign] -= 1
                        print("❌ Failed (not enough frames), retrying...")
                
                print(f"✅ Completed {sign} with {counts[sign]} samples!")
            
            print(f"\n🎉 All signs recorded successfully!")
            break
        
        try:
            idx = int(user_input)
            if idx not in SIGNS:
                print(f"Invalid. Choose 0-{len(SIGNS)-1}")
                continue
            
            sign = SIGNS[idx]
            counts[sign] += 1
            
            print(f"\n🎬 Recording: {sign} (#{counts[sign]}/{recorder.target_samples})")
            seq = recorder.record_sequence(sign, duration=3)
            if len(seq) > 5:
                recorder.save(seq, sign, counts[sign])
            else:
                counts[sign] -= 1
                print("❌ Failed (not enough frames)")
                
        except ValueError:
            print("❌ Invalid command. Type 'auto', 'auto X', 'list', or 'quit'")

if __name__ == "__main__":
    main()