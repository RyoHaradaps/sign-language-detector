import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

PROCESSED_DIR = 'data/processed'

# Load data
X_train = np.load(os.path.join(PROCESSED_DIR, 'X_train_isign.npy'))
X_val = np.load(os.path.join(PROCESSED_DIR, 'X_val_isign.npy'))
X_test = np.load(os.path.join(PROCESSED_DIR, 'X_test_isign.npy'))
y_train = np.load(os.path.join(PROCESSED_DIR, 'y_train_isign.npy'))
y_val = np.load(os.path.join(PROCESSED_DIR, 'y_val_isign.npy'))
y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test_isign.npy'))

num_classes = y_train.shape[1]

print(f"🚀 Training on iSign Dataset")
print(f"  Train: {X_train.shape}")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")
print(f"  Classes: {num_classes}")

# Build model
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
checkpoint = ModelCheckpoint('models/isign_best_model.keras', monitor='val_accuracy', save_best_only=True)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test accuracy: {test_acc:.4f}")

# Save final model
model.save('models/isign_final_model.keras')
print("✅ Model saved to models/isign_final_model.keras")

# Plot
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()
plt.savefig('demo/isign_training_curves.png')
plt.show()