# Twitter Sentiment Analysis — PySpark + NLP + Real-Time Streaming

A distributed NLP pipeline for classifying Twitter sentiment (positive/negative) using Apache Spark, scikit-learn, and real-time tweet streaming. Built as an academic project (CS644 — Big Data) combining batch ML classification with live streaming ingestion.

## Tech Stack

| Area | Tools |
|------|-------|
| Distributed Computing | Apache Spark (PySpark), SparkSQL |
| NLP / ML | scikit-learn, NLTK, TF-IDF, SVM |
| Real-Time Streaming | PySpark Structured Streaming |
| Data | Sentiment140 dataset (1.6M tweets) |
| Language | Python |

## Files

| File | Description |
|------|-------------|
| `main.py` | PySpark batch pipeline — loads train/test CSV with SparkSQL, renames and cleans columns, prepares data for distributed processing |
| `ml.py` | scikit-learn ML pipeline — TF-IDF vectorization with NLTK tokenizer + SVM classifier with 5-fold cross-validation |
| `streaming.py` | PySpark Structured Streaming pipeline — ingests live tweet stream and applies sentiment classification in real time |
| `streaming2.py` | Alternative streaming pipeline with different ingestion configuration |
| `trainingandtestdata/` | Sentiment140 train and test CSV files |
| `PROJECT REPORT CS644.pdf` | Full academic project report |

## How It Works

### Batch Classification (`ml.py`)
1. Loads 10,000 labeled tweets from the Sentiment140 dataset
2. Tokenizes text using NLTK word tokenizer
3. Applies **CountVectorizer** → **TF-IDF** transformation
4. Trains a **linear SVM** classifier
5. Evaluates with 5-fold cross-validation accuracy

### Distributed Batch (`main.py`)
1. Initializes a local Spark cluster with SparkSession
2. Loads train and test CSVs into Spark DataFrames
3. Renames columns (Polarity, ID, Text), drops unused fields
4. Handles nulls and prepares data for large-scale processing

### Real-Time Streaming (`streaming.py`)
- Uses PySpark Structured Streaming to consume a live tweet feed
- Applies the trained sentiment model to incoming records in near real-time

## Dataset

**Sentiment140** — 1.6 million tweets labeled as positive (4) or negative (0), collected via the Twitter API.  
Columns: `Polarity`, `ID`, `Date`, `Query`, `User`, `Text`

## Author

Gkeri Pepelasi
