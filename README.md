# Twitter Sentiment Analysis 🌟

A Natural Language Processing (NLP) project for analyzing sentiment in tweets using machine learning.

## Overview

This project implements sentiment analysis on Twitter data using Natural Language Processing techniques. It processes and analyzes tweets to determine whether they express positive or negative sentiment, demonstrating the practical application of NLP in social media analysis.

## Features

- Text preprocessing and cleaning of Twitter data
- Sentiment classification using Logistic Regression
- ~78% accuracy on test data
- Support for processing large datasets (1.6M tweets)
- Model persistence for future use

## Dataset

The project uses the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) from Kaggle, which contains 1.6 million tweets labeled with sentiment (positive/negative).

Key dataset characteristics:
- 1.6 million tweets
- Binary sentiment classification (0 = negative, 1 = positive)
- Balanced dataset with equal distribution of positive and negative tweets

## Technical Implementation

### Dependencies

```
python>=3.9
numpy
pandas
scikit-learn
nltk
pickle
```

### Process Flow

1. **Data Loading and Initial Processing**
   - Load the dataset using pandas
   - Handle encoding issues (using 'ISO-8859-1')
   - Rename columns for better readability
   - Convert sentiment labels (4 to 1 for positive sentiment)

2. **Text Preprocessing Pipeline**
   - Implement stemming using NLTK's SnowballStemmer
   - Remove stopwords using NLTK's stopword list
   - Convert text to lowercase
   - Create a complete text processing function handling:
     - Special characters
     - URLs
     - Twitter handles
     - Extra spaces

3. **Feature Engineering**
   - Convert text to numerical features using TF-IDF Vectorization
   - Parameters used:
     - Default n-gram range (1,1)
     - Remove English stop words
     - Strip accents
     - Lowercase transformation

4. **Model Training**
   - Split data: 80% training, 20% testing
   - Use stratified sampling to maintain class distribution
   - Train Logistic Regression with:
     - max_iterations: 1000
     - Other default scikit-learn parameters

5. **Model Evaluation**
   - Training Accuracy: ~81%
   - Testing Accuracy: ~78%
   - Model saved using pickle for future use

### Key Code Components

```python
# Text Processing Function
def stemming(content):
    stemmer = nltk.SnowballStemmer('english')
    stemmed_content = content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [stemmer.stem(word) for word in stemmed_content 
                      if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Feature Extraction
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Model Evaluation
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
```

## Future Improvements

- Implement advanced NLP techniques (BERT, Transformers)
- Add support for multi-class sentiment classification
- Create a web interface for real-time sentiment analysis
- Improve accuracy through model ensemble techniques
- Add support for multiple languages

## Acknowledgements

- [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- NLTK Documentation
- scikit-learn Documentation
- Kaggle Community
