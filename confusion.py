import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Load data
data_train = pd.read_csv("train_angle.csv")
data_test = pd.read_csv("test_angle.csv")

# Prepare training data
X_train, Y_train = data_train.iloc[:, :-1].values, pd.get_dummies(data_train['target']).values
X_test, Y_test = data_test.iloc[:, :-1].values, pd.get_dummies(data_test['target']).values

# Define a deep learning model
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(Y_train.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, Y_train, epochs=150, batch_size=8, validation_split=0.2)

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Generate predictions for the test set
Y_pred = model.predict(X_test)
Y_pred_classes = Y_pred.argmax(axis=1)  # Convert predictions to class labels
Y_true_classes = Y_test.argmax(axis=1)  # Convert one-hot labels to class labels

# Create confusion matrix
conf_matrix = confusion_matrix(Y_true_classes, Y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=data_test['target'].unique(), yticklabels=data_test['target'].unique())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print("Classification Report:")
print(classification_report(Y_true_classes, Y_pred_classes))

# Save the model
# model.save("trained_angle_model.h5")
print("Model saved as 'trained_angle_model.h5'")