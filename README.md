# News Article Analyzer

News Article Analyzer is an end-to-end Natural Language Processing (NLP) project that automatically classifies news articles into predefined categories while extracting meaningful insights from the article. The project combines classical machine learning techniques with modern NLP libraries to build a complete news analysis pipeline.

The application performs news category classification using TF-IDF and Logistic Regression, identifies important entities using spaCy Named Entity Recognition, analyzes entity frequency, and determines the sentiment of the article using VADER.

---

## Features

News category classification using Logistic Regression

Text preprocessing including tokenization, stopword removal, and lemmatization

Feature extraction using Bag of Words and TF-IDF

Named Entity Recognition using spaCy

Entity Frequency Analysis

Sentiment Analysis using VADER

Word Cloud generation

Confusion Matrix and Classification Report

Clean object-oriented project structure

---

## Categories

The model predicts one of the following news categories.

| Label | Category |
|-------|----------|
| 1 | World |
| 2 | Sports |
| 3 | Business |
| 4 | Science/Technology |

---

## Project Structure

```
NEWS_ARTICLE_ANALYZER
│
├── src
│   ├── baseline_model.py
│   ├── feature_extraction.py
│   ├── ner.py
│   ├── preprocess.py
│   ├── sentiment.py
│   └── utils.py
│
├── downloads.py
├── main.py
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## Technologies Used

Python

Pandas

NumPy

NLTK

spaCy

Scikit-learn

Matplotlib

WordCloud

Joblib

VADER Sentiment

---

## Workflow

The project follows the following pipeline.

```
Raw Dataset
      │
      ▼
Load Dataset
      │
      ▼
Text Preprocessing
      │
      ▼
Bag of Words
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Logistic Regression Training
      │
      ▼
Model Evaluation
      │
      ▼
Save Model
      │
      ▼
Analyze New Article
      │
      ├────────────► Category Prediction
      │
      ├────────────► Named Entity Recognition
      │
      ├────────────► Entity Frequency Analysis
      │
      └────────────► Sentiment Analysis
```

---

## Text Preprocessing

Every news article goes through multiple preprocessing steps before feature extraction.

Convert text to lowercase

Remove punctuation

Remove numbers

Tokenize words

Remove stopwords

Apply lemmatization

Generate clean text

---

## Feature Extraction

Two classical NLP techniques are implemented.

Bag of Words

Creates a vocabulary from the training dataset and converts each article into word frequency vectors.

TF-IDF

Assigns importance to words by reducing the influence of frequently occurring words while increasing the weight of informative words.

---

## Machine Learning Model

Algorithm

Logistic Regression

Input

TF-IDF feature vectors

Output

News category prediction

Evaluation Metrics

Accuracy Score

Classification Report

Confusion Matrix

---

## Named Entity Recognition

The project uses spaCy's pre-trained Named Entity Recognition model to identify important entities from the news article.

Examples of extracted entities include

Person

Organization

Location

Date

Money

Percentage

Product

Event

---

## Entity Frequency Analysis

After entity extraction, the frequency of each entity is calculated to determine which people, organizations, or locations dominate the article.

Example

```
Iran       : 9

US         : 8

Trump      : 7

Tuesday    : 5
```

---

## Sentiment Analysis

The project uses the VADER sentiment analyzer.

The analyzer returns

Positive score

Negative score

Neutral score

Compound score

The final sentiment is classified as

Positive

Negative

Neutral

---

## Installation

Clone the repository

```bash
git clone https://github.com/aryangaikwadgit/News_article_analyzer.git
```

Move into the project directory

```bash
cd News_article_analyzer
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

Windows

```bash
venv\Scripts\activate
```

Linux or macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Download required NLP resources

```bash
python downloads.py
```

---

## Dataset

The project uses the AG News Classification Dataset.

Download the dataset from Kaggle and place the files inside

```
data/raw/
```

The dataset should contain

```
train.csv

test.csv
```

---

## Running the Project

Train the model

```bash
python main.py
```

After training, enter a news article when prompted.

The application returns

News Category

Named Entities

Entity Frequency

Sentiment Score

Overall Sentiment

---

## Sample Output

```
Category

World

Named Entities

Iran
US
Trump
Hormuz

Top Entities

Iran : 9

US : 8

Trump : 7

Sentiment

Negative

Compound Score

-0.9977
```

---

## Future Improvements

Deep Learning based text classification

Transformer based models such as BERT

Sentence Transformers

Document Similarity Search

Vector Database Integration

Retrieval-Augmented Generation (RAG)

FastAPI backend

Interactive web interface

---

## Author

Aryan Gaikwad

GitHub

https://github.com/aryangaikwadgit
