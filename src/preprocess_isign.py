import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from collections import Counter

# Configuration
MAX_SEQ_LEN = 60
FEATURE_DIM = 126
DATA_DIR = 'data/raw_landmarks'
PROCESSED_DIR = 'data/processed'

def load_landmark_files(data_dir):
    """Load all recorded CSV files"""
    sequences = []
    labels = []
    
    if not os.path.exists(data_dir):
        print(f"❌ Directory {data_dir} not found")
        return sequences, labels
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"📂 Found {len(files)} CSV files")
    
    for file in files:
        df = pd.read_csv(os.path.join(data_dir, file))
        if 'label' in df.columns:
            label = df['label'].iloc[0]
            seq = df.drop('label', axis=1).values.astype(np.float32)
            sequences.append(seq)
            labels.append(label)
    
    return sequences, labels

def pad_sequence(seq, max_len=MAX_SEQ_LEN):
    """Pad or truncate sequence to max_len frames"""
    if len(seq) >= max_len:
        return seq[:max_len]
    pad_width = max_len - len(seq)
    return np.vstack([seq, np.zeros((pad_width, seq.shape[1]))])

def normalize_landmarks(seq):
    """Normalize landmarks to [0,1] range per frame"""
    for i in range(seq.shape[0]):
        frame = seq[i]
        if np.max(frame) - np.min(frame) > 1e-6:
            seq[i] = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
    return seq

def main():
    print("="*60)
    print("   PREPROCESSING iSign DATASET")
    print("="*60)
    
    # Load data
    print("\n📊 Loading landmark files...")
    sequences, labels = load_landmark_files(DATA_DIR)
    print(f"   Loaded {len(sequences)} sequences")
    
    if len(sequences) == 0:
        print("❌ No data found. Run record_isign_samples.py first!")
        return
    
    # Show class distribution
    label_counts = Counter(labels)
    print(f"\n📊 Class Distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"   {label}: {count}")
    
    # Pad sequences
    print(f"\n🔄 Padding sequences to {MAX_SEQ_LEN} frames...")
    padded = [pad_sequence(seq) for seq in sequences]
    X = np.array(padded)
    
    # Normalize
    print("🔄 Normalizing landmarks...")
    X = np.array([normalize_landmarks(seq) for seq in padded])
    
    # Encode labels
    print("🔄 Encoding labels...")
    le = LabelEncoder()
    y_int = le.fit_transform(labels)
    num_classes = len(le.classes_)
    y = to_categorical(y_int, num_classes)
    
    print(f"\n📊 Final Dataset:")
    print(f"   Shape: {X.shape}")
    print(f"   Classes: {num_classes}")
    print(f"   Labels: {list(le.classes_)}")
    
    # Split
    print("\n🔄 Splitting into train/val/test...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Val:   {X_val.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")
    
    # Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, 'X_train_isign.npy'), X_train)
    np.save(os.path.join(PROCESSED_DIR, 'X_val_isign.npy'), X_val)
    np.save(os.path.join(PROCESSED_DIR, 'X_test_isign.npy'), X_test)
    np.save(os.path.join(PROCESSED_DIR, 'y_train_isign.npy'), y_train)
    np.save(os.path.join(PROCESSED_DIR, 'y_val_isign.npy'), y_val)
    np.save(os.path.join(PROCESSED_DIR, 'y_test_isign.npy'), y_test)
    
    with open(os.path.join(PROCESSED_DIR, 'classes_isign.txt'), 'w') as f:
        for cls in le.classes_:
            f.write(f"{cls}\n")
    
    print("\n✅ Preprocessing complete!")
    print(f"   Saved to: {PROCESSED_DIR}")

if __name__ == "__main__":
    main()