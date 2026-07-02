# 🔐 Phishing URL Detection System

> An end-to-end machine learning system that detects malicious URLs in real time using a trained Random Forest classifier served via a Flask REST API.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![REST API](https://img.shields.io/badge/REST-API-009688?style=flat)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [System Architecture](#-system-architecture)
- [Feature Engineering](#-feature-engineering)
- [Dataset](#-dataset)
- [Models & Results](#-models--results)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [API Usage](#-api-usage)
- [Tech Stack](#-tech-stack)

---

## 📖 Overview

Phishing attacks are one of the most common cybersecurity threats, tricking users into visiting malicious websites that mimic legitimate ones. This project presents an automated **Phishing URL Detection System** that:

- Extracts **11 lexical and structural features** from any raw URL
- Classifies it as **Legitimate** or **Phishing** using a trained ML model
- Returns the prediction along with a **confidence score** via a REST API
- Requires **no external API calls** — fully self-contained feature extraction using Python's built-in `re` and `urlparse` libraries

---

## ⚙️ How It Works

```
User sends URL
      ↓
Flask /predict endpoint receives JSON request
      ↓
11 features extracted from URL structure
(length, dots, hyphens, IP address, keywords...)
      ↓
Features passed to trained Random Forest model
      ↓
Model returns: Legitimate / Phishing + confidence score
      ↓
JSON response sent back to user
```

---

## 🏗️ System Architecture

```
phishing-url-detection/
├── feature_extractor.py       # Extracts 11 URL features
├── train_model.py             # Data cleaning, EDA, model training
├── app.py                     # Flask REST API server
├── phishing_model.pkl         # Serialized trained model (joblib)
├── web-page-phishing.csv      # Raw dataset
├── cleaned_web-page-phishing.csv  # Preprocessed dataset
└── requirements.txt
```

**Request → Response flow:**

```
POST /predict
{
  "url": "http://secure-login.fakebank.com/verify?id=123"
}

→ Response:
{
  "url": "http://secure-login.fakebank.com/verify?id=123",
  "prediction": "Phishing",
  "confidence_score": 0.94
}
```

---

## 🛠️ Feature Engineering

The system extracts **11 hand-crafted lexical features** from the raw URL string — no third-party API or DNS lookup required:

| # | Feature | Why It Matters |
|---|---------|----------------|
| 1 | `url_length` | Phishing URLs are often abnormally long to hide the real domain |
| 2 | `hostname_length` | Long hostnames are a known phishing signal |
| 3 | `count_dot` | Multiple subdomains (e.g. secure.login.bank.fake.com) = red flag |
| 4 | `count_hyphen` | Legitimate domains rarely use hyphens; phishing uses them to mimic brands |
| 5 | `count_at` | Browser ignores everything before `@` — classic phishing trick |
| 6 | `count_slash` | Unusually deep URL paths indicate obfuscation |
| 7 | `count_question` | Excessive query parameters are suspicious |
| 8 | `count_equal` | Many `=` signs = many parameters = possible redirect manipulation |
| 9 | `count_digit` | High digit density in URL path is a known phishing pattern |
| 10 | `contains_suspicious_word` | Words like `secure`, `login`, `verify`, `bank`, `update` build false trust |
| 11 | `contains_ip` | Legitimate sites use domain names, not raw IPs — IP = strong phishing signal |

---

## 📊 Dataset

**Source:** `web-page-phishing.csv` (Kaggle / public phishing URL dataset)

**Preprocessing pipeline applied:**
- Removed null values (`dropna`)
- Removed duplicate records (`drop_duplicates`)
- Auto-detected label column (`status` / `label` / `phishing` / `class`)
- Applied `LabelEncoder` for string labels → binary (0 = Legitimate, 1 = Phishing)
- Saved cleaned dataset as `cleaned_web-page-phishing.csv`

---

## 📈 Models & Results

Three classifiers were trained and evaluated on an **80/20 stratified train-test split:**

| Model | Accuracy |
|-------|----------|
| Logistic Regression | Baseline |
| XGBoost | High |
| **Random Forest** ✅ | **96%** |

**Random Forest** was selected as the final model for deployment due to its highest accuracy and robustness on the phishing dataset.

**Evaluation metrics used:**
- Accuracy Score
- Precision, Recall, F1-Score (per class)
- Classification Report (via `sklearn.metrics`)

The trained model is serialized using `joblib` for lightweight, retrainable deployment without retraining overhead.

---

## 📁 Project Structure

```
phishing-url-detection/
│
├── app.py                        # Flask REST API — /predict endpoint
├── train_model.py                # EDA, preprocessing, model training
├── feature_extractor.py          # 11-feature URL extraction pipeline
├── phishing_model.pkl            # Saved Random Forest model
│
├── web-page-phishing.csv         # Raw dataset
├── cleaned_web-page-phishing.csv # Cleaned & encoded dataset
│
└── requirements.txt              # Python dependencies
```

---

## 🚀 Installation & Setup

**1. Clone the repository:**
```bash
git clone https://github.com/subha-btech805/phishing-url-detection.git
cd phishing-url-detection
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Train the model (generates phishing_model.pkl):**
```bash
python train_model.py
```

**4. Start the Flask API server:**
```bash
python app.py
```

Server runs at: `http://localhost:5000`

---

## 🔌 API Usage

**Endpoint:** `POST /predict`

**Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "http://secure-login.fakebank.com/verify?id=123"}'
```

**Response:**
```json
{
  "url": "http://secure-login.fakebank.com/verify?id=123",
  "prediction": "Phishing",
  "confidence_score": 0.94
}
```

**Test with a legitimate URL:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.google.com"}'
```

```json
{
  "url": "https://www.google.com",
  "prediction": "Legitimate",
  "confidence_score": 0.98
}
```

---

## 🧰 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| ML Framework | Scikit-learn, XGBoost |
| API Framework | Flask |
| Data Processing | Pandas, NumPy |
| Feature Extraction | Python `re`, `urllib.parse` |
| Model Serialization | joblib |
| Dataset | web-page-phishing.csv |

---

## 👩‍💻 Author

**Subha M**
B.Tech Artificial Intelligence & Data Science
VSB College of Engineering, Coimbatore

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/subha-meenakshi-sundaram-10ab2a30b/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/subha-btech805)

---

> ⭐ If you found this project useful, consider giving it a star!
