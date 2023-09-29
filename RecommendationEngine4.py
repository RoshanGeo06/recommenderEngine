# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.neighbors import NearestNeighbors

# Step 1: Data Preparation
df = pd.read_excel('syntheticDataset.xlsx')  # replace with your file path

# Step 2: Data Preprocessing
df = df.drop(columns=['Classes'])  # remove the 'Classes' column

# Step 3: Implementing Autoencoder
ncol = df.shape[1]  # number of columns in the dataframe
input_dim = Input(shape=(ncol,))

# Define the encoder layers
encoded1 = Dense(100, activation='relu')(input_dim)
encoded2 = Dense(50, activation='relu')(encoded1)

# Define the decoder layers
decoded1 = Dense(50, activation='relu')(encoded2)
decoded2 = Dense(ncol, activation='sigmoid')(decoded1)

# Combine encoder and decoder into an autoencoder model
autoencoder = Model(inputs=input_dim, outputs=decoded2)

# Compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Step 4: Train Autoencoder
x_train, x_test = train_test_split(df, test_size=0.2, random_state=42)
autoencoder.fit(x_train, x_train, epochs=50, batch_size=100, shuffle=True, validation_data=(x_test, x_test))

# Step 5: Implementing KNN
knn = NearestNeighbors(n_neighbors=3)
knn.fit(df)

# Step 6: User Requirements
user_input = [18.04308426, 2.596643286, 1.703254638, 7.476318192, 1.10257114, 6.922332673, 2.854282121]  # replace with actual user input

# Step 7: Making Recommendations
distances, indices = knn.kneighbors([user_input], n_neighbors=3)

# Print the recommended ships
for i in range(len(indices[0])):
    print(df.iloc[indices[0][i]])
