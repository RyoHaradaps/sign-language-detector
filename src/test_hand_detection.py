import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize hand detection
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Use IP Webcam (your phone)
phone_ip = "http://192.0.0.4:8080/video"  # Update to your IP
print(f"📱 Connecting to: {phone_ip}")

cap = cv2.VideoCapture(phone_ip)

if not cap.isOpened():
    print("❌ Cannot connect to camera!")
    print("   Make sure IP Webcam is running on your phone")
    exit()

print("✅ Camera connected!")
print("📸 Show your hand to the camera")
print("   - Green dots = landmarks detected")
print("   - Lines = connections between landmarks")
print("   - Press 'q' to quit")
print("   - Press 'r' to reset detection")
print()

# Variables for FPS
fps_start = time.time()
fps_counter = 0
fps_display = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Camera disconnected")
        break
    
    # Resize for faster processing
    frame = cv2.resize(frame, (640, 480))
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # Count hands detected
    num_hands = 0
    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)
        
        # Draw landmarks on each hand
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    # Calculate FPS
    fps_counter += 1
    if time.time() - fps_start >= 1.0:
        fps_display = fps_counter
        fps_counter = 0
        fps_start = time.time()
    
    # Display information on screen
    cv2.putText(frame, f"Hands detected: {num_hands}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps_display}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Status messages
    if num_hands == 0:
        cv2.putText(frame, "🤚 Show your hand!", (200, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif num_hands == 1:
        cv2.putText(frame, "✅ One hand detected!", (200, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "✅ Both hands detected!", (200, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('Hand Detection Test', frame)
    
    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("👋 Quitting...")
        break
    elif key == ord('r'):
        print("🔄 Resetting detection...")
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("✅ Test completed!")