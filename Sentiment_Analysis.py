# TASK 4: AI-Driven Sentiment Analysis System

import pandas as pd
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------
# Create Dataset
# -------------------------------
data = {
    "text": [
        "I love this product",
        "This is the worst service",
        "It is okay, not bad",
        "Amazing experience",
        "Very disappointing",
        "I am happy with the service",
        "Not satisfied at all",
        "Average quality product"
    ],
    "sentiment": [
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative",
        "positive",
        "negative",
        "neutral"
    ]
}

df = pd.DataFrame(data)

# -------------------------------
# Text Preprocessing
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-z ]', '', text)
    return text

df["clean_text"] = df["text"].apply(clean_text)

# -------------------------------
# Feature Extraction (TF-IDF)
# -------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])
y = df["sentiment"]

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -------------------------------
# Train Sentiment Model
# -------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------
# Model Evaluation
# -------------------------------
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("\nModel Accuracy:", accuracy)

# -------------------------------
# Test on Real User Input
# -------------------------------
user_text = input("\nEnter text for sentiment analysis:\n")

clean_user = clean_text(user_text)
user_vec = vectorizer.transform([clean_user])

prediction = model.predict(user_vec)[0]
print("Predicted Sentiment:", prediction)

# -------------------------------
# Visualization
# -------------------------------
df['sentiment'].value_counts().plot(kind='bar', title='Sentiment Distribution')
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()
