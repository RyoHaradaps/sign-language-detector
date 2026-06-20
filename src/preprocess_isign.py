import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Configuration
MAX_SEQ_LEN = 60
FEATURE_DIM = 126
TARGET_SIGNS = ['namaste', 'thank_you', 'please', 'sorry', 'help', 'yes', 'no',
                'stop', 'water', 'food', 'good', 'bad', 'love', 'friend', 'work']

def load_landmark_files(data_dir='data/raw_landmarks'):
    """Load recorded CSV files"""
    sequences = []
    labels = []
    
    if not os.path.exists(data_dir):
        print(f"⚠️ Directory {data_dir} not found")
        return sequences, labels
    
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(data_dir, file))
            if 'label' in df.columns:
                label = df['label'].iloc[0]
                if label in TARGET_SIGNS:
                    seq = df.drop('label', axis=1).values.astype(np.float32)
                    sequences.append(seq)
                    labels.append(label)
    
    return sequences, labels

def pad_sequence(seq, max_len=MAX_SEQ_LEN):
    if len(seq) >= max_len:
        return seq[:max_len]
    pad_width = max_len - len(seq)
    return np.vstack([seq, np.zeros((pad_width, seq.shape[1]))])

def main():
    print("📊 Loading iSign filtered dataset...")
    
    # Load recorded landmarks
    sequences, labels = load_landmark_files()
    print(f"  Found {len(sequences)} recorded samples")
    
    # Load iSign word frequency for context
    freq_df = pd.read_csv('data/isign/isign_filtered_dataset.csv')
    print(f"  iSign has {len(freq_df):,} sentences with ISL words")
    
    if len(sequences) == 0:
        print("❌ No recorded samples found. Please run record_isign_samples.py first.")
        return
    
    # Pad sequences
    X = np.array([pad_sequence(seq) for seq in sequences])
    
    # Encode labels
    le = LabelEncoder()
    y_int = le.fit_transform(labels)
    y = to_categorical(y_int, num_classes=len(le.classes_))
    
    print(f"📊 Dataset shape: {X.shape}")
    print(f"📊 Classes: {le.classes_}")
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Save
    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    np.save(os.path.join(processed_dir, 'X_train_isign.npy'), X_train)
    np.save(os.path.join(processed_dir, 'X_val_isign.npy'), X_val)
    np.save(os.path.join(processed_dir, 'X_test_isign.npy'), X_test)
    np.save(os.path.join(processed_dir, 'y_train_isign.npy'), y_train)
    np.save(os.path.join(processed_dir, 'y_val_isign.npy'), y_val)
    np.save(os.path.join(processed_dir, 'y_test_isign.npy'), y_test)
    
    with open(os.path.join(processed_dir, 'classes_isign.txt'), 'w') as f:
        for cls in le.classes_:
            f.write(f"{cls}\n")
    
    print("✅ Preprocessing complete!")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")

if __name__ == "__main__":
    main()