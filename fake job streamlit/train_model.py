import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# -------------------------------
# Load Dataset (No Full Path)
# -------------------------------
df = pd.read_excel("fake_real_job_postings_3000x25.xlsx")

print("Dataset Loaded")
print("Columns:", df.columns)


# -------------------------------
# Fill Missing Values
# -------------------------------
df = df.fillna("")


# -------------------------------
# Prepare Text Columns Safely
# -------------------------------
text_columns = [
    "job_title",
    "description",
    "requirements",
    "benefits",
    "company_profile"
]

# If any column missing, create empty
for col in text_columns:
    if col not in df.columns:
        df[col] = ""


# Combine text
df["text"] = df[text_columns].agg(" ".join, axis=1)


# -------------------------------
# Target Column
# -------------------------------
# 1 = Fake, 0 = Real
y = df["fraud_reason"]
X = df["text"]


# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------------
# Vectorization
# -------------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# -------------------------------
# Train Model
# -------------------------------
model = LogisticRegression(max_iter=1000)

model.fit(X_train_vec, y_train)


# -------------------------------
# Accuracy
# -------------------------------
y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)

print("Model Accuracy:", acc)


# -------------------------------
# Save Model
# -------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)


print("Files Saved: model.pkl, vectorizer.pkl")
