import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load your dataset (replace 'syntheticDataset.xlsx' with your Excel file)
data = pd.read_excel('syntheticDataset.xlsx')

# Preprocess the data (you may need to handle missing values and encode the 'Classes' column)
# For simplicity, we assume the data is preprocessed and ready for modeling.

# Split the data into training and testing sets (80% training, 20% testing)
from sklearn.model_selection import train_test_split

X = data[['LOA', 'LB', 'LS', 'BD', 'DD', 'BS', 'WL']]
y = data['Classes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# User input for ship requirements
user_input = [18.04308426, 2.596643286, 1.703254638, 7.476318192, 1.10257114, 6.922332673, 2.854282121]  # Replace with user's input

# Predict the ship class based on user input
predicted_class = clf.predict([user_input])

recommended_ships = data[data['Classes'] == predicted_class[0]]
print("Recommended Ships:")
print(recommended_ships)

# Optional: Evaluate the model's performance on the test set
from sklearn.metrics import accuracy_score, classification_report

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

report = classification_report(y_test, y_pred)
print("\nClassification Report:\n", report)
