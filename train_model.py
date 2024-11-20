import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
cancer = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv')

# Preprocess the dataset
y = cancer['diagnosis']
X = cancer.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2529)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'cancer_prediction_model.pkl')
print("Model trained and saved successfully!")
