import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from sklearn.neighbors import NearestNeighbors

# Load the dataset
df = pd.read_excel('syntheticDataset.xlsx', header=0)

# Let's assume that the last column of your dataset is the class
# We will drop it for now and handle it separately
classes = df.pop('Classes')

# Normalize the dataset
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Define the encoding dimension
encoding_dim = 32

# Define the input shape
input_shape = df.shape[1]

# Define the encoder
input_layer = Input(shape=(input_shape,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)

# Define the decoder
decoded = Dense(input_shape, activation='sigmoid')(encoded)

# Define the autoencoder model
autoencoder = Model(input_layer, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
autoencoder.fit(df, df, epochs=50, batch_size=256, shuffle=True)

# Use the encoder to transform the data
encoder = Model(input_layer, encoded)
encoded_data = encoder.predict(df)

# Train the KNN model
knn = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(encoded_data)

# Now you can use the trained KNN model to make recommendations
# For example, let's find the 3 items most similar to the first item in our dataset
distances, indices = knn.kneighbors(encoded_data[0].reshape(1, -1))
recommended_classes = classes[indices[0]]

print(recommended_classes)
