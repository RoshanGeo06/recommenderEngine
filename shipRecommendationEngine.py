import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from sklearn.neighbors import NearestNeighbors

# Load the dataset
df = pd.read_excel('syntheticDataset.xlsx', header=0)

# Normalize the numeric columns in the dataset
numeric_columns = df.select_dtypes(include=['number']).columns
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Define the encoding dimension
encoding_dim = 32

# Define the input shape
input_shape = len(numeric_columns)

# Define the encoder
input_layer = Input(shape=(input_shape,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)

# Define the decoder
decoded = Dense(input_shape, activation='sigmoid')(encoded)

# Define the autoencoder model
autoencoder = Model(input_layer, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')  # Use mean squared error loss

# Train the model
autoencoder.fit(df[numeric_columns], df[numeric_columns], epochs=50, batch_size=256, shuffle=True)

# Use the encoder to transform the data
encoder = Model(input_layer, encoded)

# Get user input (for example, you can use input())
user_input = pd.DataFrame(columns=numeric_columns)
for column in numeric_columns:
    value = float(input(f"Enter a value for {column}: "))
    user_input[column] = [value]

# Normalize user input using the same scaler
user_input = scaler.transform(user_input)

# Encode user input
encoded_user_input = encoder.predict(user_input)

# Train the KNN model on the encoded data
knn = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(encoder.predict(df[numeric_columns]))

# Find nearest neighbors to the user's input
distances, indices = knn.kneighbors(encoded_user_input)

# Print recommendations based on nearest neighbors
print("Recommendations based on your input:")
for i in indices[0]:
    print(df.iloc[i])  # Display the details of recommended items
