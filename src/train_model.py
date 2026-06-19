import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Attention, Flatten, Input, Permute, Multiply, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Paths
PROCESSED_DIR = "/mnt/c/Users/surya/projects/sign_language_detector/data/processed"
MODEL_DIR = r"C:\Users\surya\projects\sign_language_detector\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load preprocessed data
X_train = np.load(os.path.join(PROCESSED_DIR, 'X_train.npy'))
X_val = np.load(os.path.join(PROCESSED_DIR, 'X_val.npy'))
X_test = np.load(os.path.join(PROCESSED_DIR, 'X_test.npy'))
y_train = np.load(os.path.join(PROCESSED_DIR, 'y_train.npy'))
y_val = np.load(os.path.join(PROCESSED_DIR, 'y_val.npy'))
y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'))

# Load class names
with open(os.path.join(PROCESSED_DIR, 'classes.txt'), 'r') as f:
    classes = [line.strip() for line in f.readlines()]
num_classes = len(classes)

print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
print(f"Classes: {classes}")

# Build LSTM + Attention model
def build_model(input_shape, num_classes):
    model = Sequential()
    
    # Bidirectional LSTM layer 1
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    
    # Attention mechanism (simple multiplicative attention)
    # We'll use a custom attention layer
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.3))
    
    # Attention layer: compute weights per timestep
    attention = Dense(1, activation='tanh')(model.output)
    attention = Flatten()(attention)
    attention = Dense(input_shape[0], activation='softmax')(attention)
    attention = Lambda(lambda x: x[0] * x[1][:, :, None])([attention, model.output])
    attention = Lambda(lambda x: tf.reduce_sum(x, axis=1))(attention)
    
    # Alternatively, use TensorFlow's built-in attention (simpler)
    # For simplicity, we'll use a GlobalAveragePooling1D after the BiLSTM
    # But let's implement a proper attention layer
    
    # Actually, easier: use a Dense layer to compute attention weights
    # Let me provide a cleaner version using `Attention` layer from Keras
    
    # Better approach (works with Keras):
    model.add(Attention())  # This will attend to the outputs of the last LSTM
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

# Simpler but effective model (without custom attention complexity)
def build_simple_model(input_shape, num_classes):
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=False)),  # return_sequences=False for final output
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Build the model
input_shape = (X_train.shape[1], X_train.shape[2])  # (60, 126)
model = build_simple_model(input_shape, num_classes)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, 'best_model.h5'), 
                             monitor='val_accuracy', save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

# Predict
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=classes))

# Confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))
plt.show()

# Plot training curves
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(MODEL_DIR, 'training_curves.png'))
plt.show()

print(f"Model saved to {os.path.join(MODEL_DIR, 'best_model.h5')}")