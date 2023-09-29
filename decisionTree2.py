import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import numpy as np

# Step 1: Data Preparation
# Load the Excel sheet into a Pandas DataFrame
df = pd.read_excel("syntheticDataset.xlsx")

# Encode the "Classes" column since decision trees require numeric input
label_encoder = LabelEncoder()
df["Classes"] = label_encoder.fit_transform(df["Classes"])

# Split the dataset into features (X) and target (y)
X = df.drop(columns=["Classes"])
y = df["Classes"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Building the Decision Tree Model
# Create a Decision Tree Classifier
decision_tree = DecisionTreeClassifier()

# Train the model on the training data
decision_tree.fit(X_train, y_train)

# Make predictions on the test data
y_pred = decision_tree.predict(X_test)

# Step 3: Recommendation Generation
# Collect user requirements (you can replace these with user inputs)
user_loa = 23.44869885
user_lb = 1.586849978
user_ls = 2.759827071
user_bd = 6.735263184
user_dd = 1.635237083
user_bs = 9.697277712
user_wl = 3.877318069

# Prepare the user's input as a Pandas DataFrame
user_input = pd.DataFrame({
    'LOA': [user_loa],
    'LB': [user_lb],
    'LS': [user_ls],
    'BD': [user_bd],
    'DD': [user_dd],
    'BS': [user_bs],
    'WL': [user_wl]
})

# Calculate the MSE between user input and all ships in the dataset
similarities = []
for index, row in df.iterrows():
    mse = mean_squared_error(user_input.values[0], row.values[:-1])  # Compare each feature separately
    similarities.append((index, mse))

# Sort the ships by similarity (in ascending order)
similarities.sort(key=lambda x: x[1])

# Select the top 3 closest matches
top_3_ships = similarities[:3]

# Print the details of the top 3 closest ships
print("Top 3 ships closest to your requirements:")
for idx, _ in top_3_ships:
    ship_details = df.iloc[idx]
    print(ship_details)

# Step 4: Display the Recommendation
user_prediction = decision_tree.predict(user_input)
user_recommendation = label_encoder.inverse_transform(user_prediction)
print(f"\nBased on your requirements, we recommend a ship of class: {user_recommendation[0]}")
