import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import keras
from keras.models import Sequential
from keras.layers import Dense

# Load data
df = pd.read_csv('/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/App/cancer patient.csv')

# Map the levels to numerical values
level_mapping = {'Low': 0.0, 'Medium': 1.0, 'High': 2.0}
df['Level'] = df['Level'].replace(level_mapping)

# Select features and target
X = df.iloc[:, 2:-1].values  # Select features
y = df['Level'].values       # Select target

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initial model to calculate feature importance
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X.shape[1]))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))  # Use softmax for multi-class classification

# Compile the model with a new optimizer instance
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model for a few epochs to calculate feature importance
model.fit(X_train, keras.utils.to_categorical(y_train, 3), epochs=5, batch_size=32, validation_data=(X_test, keras.utils.to_categorical(y_test, 3)), verbose=2)

# Calculate baseline accuracy
baseline_accuracy = accuracy_score(y_test, np.argmax(model.predict(X_test), axis=1))

# Permutation feature importance
feature_importances = np.zeros(X.shape[1])
for i in range(X.shape[1]):
    X_permuted = X_test.copy()
    X_permuted[:, i] = shuffle(X_permuted[:, i])
    permuted_accuracy = accuracy_score(y_test, np.argmax(model.predict(X_permuted), axis=1))
    feature_importances[i] = baseline_accuracy - permuted_accuracy

# Get top 10 important features
feature_names = df.columns[2:-1]
feature_importance_dict = dict(zip(feature_names, feature_importances))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
top_features = [x[0] for x in sorted_features[:10]]
print("Top 10 important features:", top_features)

# Filter data to only use the top 10 important features
X = df[top_features].values
X = scaler.fit_transform(X)  # Re-scale with only top features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a new model using only the top features
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=len(top_features)))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))  # Use softmax for multi-class classification

# Compile the model with a new optimizer instance
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, keras.utils.to_categorical(y_train, 3), epochs=10, batch_size=32, validation_data=(X_test, keras.utils.to_categorical(y_test, 3)), verbose=2)

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, keras.utils.to_categorical(y_test, 3))
print('Test accuracy:', round(accuracy * 100, 2), '%')

# Plot the training and validation loss and accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Save the model and scaler with the top features
model.save('/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/App/my_model.h5')
filename = 'scaler.pkl'
with open(filename, 'wb') as f:
    pickle.dump(scaler, f)
