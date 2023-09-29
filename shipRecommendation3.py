import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from sklearn.neighbors import NearestNeighbors

# Load the dataset
df = pd.read_excel('syntheticDataset.xlsx')

# Extract the target variable 'Classes'
classes = df['Classes']

# Drop the 'Classes' column to keep only the feature columns
df = df.drop(columns=['Classes'])

# Normalize the feature columns
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

# Step 1: Collect User Preferences (example)
user_preferences = pd.DataFrame(
    {'LOA': [23.44869885], 'LB': [1.586849978], 'LS': [2.759827071], 'BD': [6.735263184], 'DD': [1.635237083], 'BS': [9.697277712], 'WL': [3.877318069]})
# Normalize user preferences
user_preferences = pd.DataFrame(scaler.transform(user_preferences), columns=user_preferences.columns)

# Step 3: Encode the User Profile
user_encoded = encoder.predict(user_preferences)

# Step 4: Find Similar Items
distances, indices = knn.kneighbors(user_encoded)

if len(indices[0]) == 0:
    # No nearest neighbors found, print all ship values
    print("No similar ships found. Here are all the ship values:")
    print(classes)
else:
    # Step 5: Provide Recommendations
    recommended_indices = indices[0]
    recommended_classes = classes.iloc[recommended_indices]
    print("Recommended Classes:", recommended_classes)

    # Print the values of all parameters for the recommended classes
    recommended_parameters = df.iloc[recommended_indices]
    print("Values of Parameters for Recommended Classes:")
    print(recommended_parameters)
