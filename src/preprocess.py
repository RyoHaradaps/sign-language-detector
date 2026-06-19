import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Configuration
DATA_DIR = r"C:\Users\surya\projects\sign_language_detector\data\raw_landmarks"
PROCESSED_DIR = r"C:\Users\surya\projects\sign_language_detector\data\processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

MAX_SEQ_LEN = 60   # Target number of frames per sample (pad/truncate)
FEATURE_DIM = 126  # 2 hands * 21 landmarks * 3 coordinates

# Load all CSV files
def load_data(data_dir):
    sequences = []
    labels = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(data_dir, file))
            # Drop the 'label' column if present (it's at the end)
            if 'label' in df.columns:
                label = df['label'].iloc[0]
                df = df.drop('label', axis=1)
            else:
                # If no label column, infer from filename (optional)
                label = file.split('_')[0]
            # Convert to numpy array
            seq = df.values.astype(np.float32)
            sequences.append(seq)
            labels.append(label)
    return sequences, labels

def pad_sequence(seq, max_len):
    """Pad or truncate sequence to max_len frames"""
    if len(seq) >= max_len:
        return seq[:max_len]
    else:
        pad_width = max_len - len(seq)
        pad = np.zeros((pad_width, seq.shape[1]))
        return np.vstack([seq, pad])

def normalize_landmarks(seq):
    """Normalize landmarks to [0,1] range per frame (optional)"""
    # Simple min-max normalization per frame (ignoring zeros)
    for i in range(seq.shape[0]):
        frame = seq[i]
        # Avoid division by zero if all zeros
        if np.max(frame) - np.min(frame) > 1e-6:
            seq[i] = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
    return seq

def main():
    print("Loading data...")
    sequences, labels = load_data(DATA_DIR)
    print(f"Loaded {len(sequences)} sequences")

    # Pad sequences
    print(f"Padding sequences to {MAX_SEQ_LEN} frames...")
    padded = [pad_sequence(seq, MAX_SEQ_LEN) for seq in sequences]
    X = np.array(padded)

    # Normalize (optional, can be done later)
    # X = np.array([normalize_landmarks(seq) for seq in padded])

    # Encode labels
    le = LabelEncoder()
    y_int = le.fit_transform(labels)
    num_classes = len(le.classes_)
    y = to_categorical(y_int, num_classes)

    print(f"Input shape: {X.shape}")  # (samples, time, features)
    print(f"Classes: {le.classes_}")

    # Train/validation/test split (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Save processed data as numpy arrays
    np.save(os.path.join(PROCESSED_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(PROCESSED_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(PROCESSED_DIR, 'y_test.npy'), y_test)

    # Save label encoder classes for later use
    with open(os.path.join(PROCESSED_DIR, 'classes.txt'), 'w') as f:
        for cls in le.classes_:
            f.write(f"{cls}\n")

    print("Preprocessing complete. Saved to:", PROCESSED_DIR)

if __name__ == "__main__":
    main()