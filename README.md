
---

```markdown
# 🧠 Fake News Detection using GRU + Word2Vec

A deep learning–based project that classifies news articles as **FAKE** or **REAL**, built using **PyTorch (GRU model)** and **Google’s Word2Vec embeddings**.  
Includes both **CLI** and **REST API (Flask)** interfaces for flexible use.

---

## ✨ Features
✅ Detects fake vs. real news based on text input  
✅ Uses pretrained **GoogleNews Word2Vec (300D)** embeddings  
✅ Flask REST API for real-time predictions  
✅ CLI mode for local/offline testing  

---

## 📁 Project Structure

```

.
├── app.py                         # Flask API server
├── predict.py                     # Text preprocessing + prediction logic
├── model.py                       # GRUClassifier architecture
├── gru_classifier.pth             # Trained PyTorch model weights
├── requirements.txt               # Dependencies
├── GoogleNews-vectors-negative300.bin.gz   # Word2Vec embeddings (not included)
└── README.md                      # Project documentation

````

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/fake-news-detector.git
cd fake-news-detector
````

### 2️⃣ Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate     # Linux / macOS
# OR
venv\Scripts\activate        # Windows
```

### 3️⃣ Install Required Packages

```bash
pip install -r requirements.txt
```

---

## 🧩 Download the Pretrained Word2Vec Model

The **GoogleNews-vectors-negative300.bin.gz** (≈1.6 GB) file is required but **not included** in this repo due to its large size.

### 📥 Download Options

#### 🔹 Option 1 — Official Source

[https://code.google.com/archive/p/word2vec/](https://code.google.com/archive/p/word2vec/)

#### 🔹 Option 2 — Command Line (if available)

```bash
wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
```

> ⚠️ If the above link no longer works, search for
> **“GoogleNews-vectors-negative300.bin.gz”** on [Kaggle](https://www.kaggle.com/) or [GitHub Releases](https://github.com/).

Once downloaded, move the file into your project root directory:

```
fake-news-detector/
│
├── GoogleNews-vectors-negative300.bin.gz
├── predict.py
├── app.py
└── ...
```

---

## 💻 Run in CLI Mode

You can test predictions directly in your terminal:

```bash
python predict.py
```

Example:

```
🧠 Loading Word2Vec...
✅ Word2Vec loaded successfully!
✅ GRU model loaded and ready for inference.

💬 Fake News Detector — type or paste text below.
Type 'q' to quit.

Enter news text: Scientists discover water on Mars.
Prediction: REAL (confidence: 0.91)
```

---

## 🌐 Run as an API Server

Run the Flask app for API-based predictions.

### 1️⃣ Start the Server

```bash
python app.py
```

You’ll see:

```
✅ Word2Vec loaded successfully!
✅ GRU model loaded and ready for inference.
🚀 Flask API running on http://0.0.0.0:5000
```

### 2️⃣ Send a POST Request

Use `curl` or Postman:

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text":"Scientists discover water on Mars."}'
```

Expected Response:

```json
{
  "label": "REAL",
  "confidence": 0.91
}
```

---

## 🧾 .gitignore Recommendation

To prevent large or unnecessary files from being committed:

```
venv/
__pycache__/
GoogleNews-vectors-negative300.bin.gz
```
