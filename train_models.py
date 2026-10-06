import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from preprocess import clean_text


def train_model(data_path, model_name):

    # Load dataset
    df = pd.read_csv(data_path)

    # Clean messages
    df['message'] = df['message'].apply(clean_text)

    # Create ML pipeline
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])

    # Train the model
    model.fit(df['message'], df['label'])

    # Save trained model
    joblib.dump(model, f"models/{model_name}.pkl")

    print(f"{model_name} trained successfully")


# Train all AI models
train_model('data/sentiment_data.csv', 'sentiment_model')

train_model('data/spam_data.csv', 'spam_model')

train_model('data/urgency_data.csv', 'urgency_model')

train_model('data/tone_data.csv', 'tone_model')