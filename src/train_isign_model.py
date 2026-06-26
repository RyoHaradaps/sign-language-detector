import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# Paths
PROCESSED_DIR = 'data/processed'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# Load data
print("="*60)
print("   TRAINING iSign MODEL")
print("="*60)

print("\n📊 Loading preprocessed data...")
X_train = np.load(os.path.join(PROCESSED_DIR, 'X_train_isign.npy'))
X_val = np.load(os.path.join(PROCESSED_DIR, 'X_val_isign.npy'))
X_test = np.load(os.path.join(PROCESSED_DIR, 'X_test_isign.npy'))
y_train = np.load(os.path.join(PROCESSED_DIR, 'y_train_isign.npy'))
y_val = np.load(os.path.join(PROCESSED_DIR, 'y_val_isign.npy'))
y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test_isign.npy'))

with open(os.path.join(PROCESSED_DIR, 'classes_isign.txt'), 'r') as f:
    classes = [line.strip() for line in f.readlines()]

num_classes = y_train.shape[1]

print(f"   Train: {X_train.shape}")
print(f"   Val:   {X_val.shape}")
print(f"   Test:  {X_test.shape}")
print(f"   Classes: {num_classes}")
print(f"   Labels: {classes}")

# Build model
print("\n🔨 Building model...")
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks (FIXED)
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Save in .h5 format for compatibility
checkpoint = ModelCheckpoint(
    os.path.join(MODELS_DIR, 'isign_best_model.h5'),  # Use .h5 instead of .keras
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train
print("\n🚀 Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

# Evaluate
print("\n📊 Evaluating model...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"   Test accuracy: {test_acc:.4f}")

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=classes))

# Confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix - iSign Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'isign_confusion_matrix.png'))
plt.show()

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'isign_training_curves.png'))
plt.show()

print(f"\n✅ Model saved to: {MODELS_DIR}")
print(f"   Best model: isign_best_model.h5")