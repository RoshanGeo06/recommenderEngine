import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Step 1: Data Preparation
df = pd.read_excel('syntheticDataset.xlsx')  # replace with your file path

# Step 2: Data Preprocessing
X = df.drop(columns=['Classes'])  # remove the 'Classes' column
y = df['Classes']

# Convert classes to categorical for the neural network
y = to_categorical(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build the Model
model = Sequential()
model.add(Dense(100, input_dim=X.shape[1], activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))  # output layer

# Step 4: Compile the Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 5: Train the Model
model.fit(X_train, y_train, epochs=50, batch_size=100)

# Step 6: Evaluate the Model
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

# Step 7: Make Predictions
user_input = [18.04308426, 2.596643286, 1.703254638, 7.476318192, 1.10257114, 6.922332673, 2.854282121]  # replace with actual user input
predicted_class = model.predict_classes([user_input])
print('Predicted Class:', predicted_class[0])
