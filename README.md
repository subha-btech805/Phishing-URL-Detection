# Phishing-URL-Detection    
With the rapid growth of internet usage, phishing attacks have become one of the most common and dangerous cybersecurity threats, targeting users to steal sensitive information such as login credentials, banking details, and personal data. This project presents a Phishing URL Detection System designed to automatically analyze and classify URLs as legitimate or malicious, thereby preventing potential security breaches. The system leverages machine learning techniques, including traditional algorithms like Random Forest and XGBoost, as well as deep learning models such as CNN, LSTM, and Transformer-based models, to detect phishing attempts accurately. It uses a combination of handcrafted URL features (like length, special characters, and shortening services) and tokenized URL patterns to improve detection performance. The system provides a confidence score indicating the likelihood of a URL being phishing, allowing users to make informed decisions. This project emphasizes real-time URL analysis, high detection accuracy, and practical usability, contributing to enhanced online security for individuals and organizations.
# Feature Extraction
import re
from urllib.parse import urlparse

def extract_features(url):
    features = {}
    
   features['url_length'] = len(url)
   features['hostname_length'] = len(urlparse(url).hostname)
    
  features['count_dot'] = url.count('.')
  features['count_hyphen'] = url.count('-')
  features['count_at'] = url.count('@')
  features['count_slash'] = url.count('/')
  features['count_question'] = url.count('?')
  features['count_equal'] = url.count('=')
   features['count_digit'] = sum(c.isdigit() for c in url)
    suspicious_words = ["secure", "account", "update", "free", "verify", "login", "bank"]
  features['contains_suspicious_word'] = int(any(word in url.lower() for word in suspicious_words))
    
    
  match_ip = re.search(r'\d+\.\d+\.\d+\.\d+', url)
   features['contains_ip'] = 1 if match_ip else 0
    
  return list(features.values())
#Models
import pandas as pd
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

path = "web-page-phishing.csv"
df = pd.read_csv(path)
print("Before Cleaning:", df.shape)

print("\nColumns found:", list(df.columns))


label_col = None
for col in df.columns:
    if col.lower() in ["status", "label", "class", "phishing", "target"]:
        label_col = col
        break

if label_col is None:
    sys.exit("Could not find label column. Rename target column to 'phishing' / 'label' / 'status'.")

print(f"\nUsing label column: {label_col}")

df = df.dropna(subset=[label_col])
df = df.dropna()
df = df.drop_duplicates()

print("\nUnique values in label before encoding:", df[label_col].unique())
if not set(df[label_col].unique()).issubset({0, 1}):
    le = LabelEncoder()
    df[label_col] = le.fit_transform(df[label_col])

df = df.rename(columns={label_col: "status"})

print("\nFinal label distribution:")
print(df["status"].value_counts())


df.to_csv("cleaned_web-page-phishing.csv", index=False)
print("\nCleaned dataset saved.")



print("\n=== TRAINING MODEL ===")

X = df.drop("status", axis=1)
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "phishing_model.pkl")
print("\nModel saved as: phishing_model.pkl")

      
