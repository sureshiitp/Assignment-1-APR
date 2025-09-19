import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df[iris.feature_names], df['target'], test_size=0.2, random_state=42
)

# Train model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("KNN Model Accuracy:", accuracy)
