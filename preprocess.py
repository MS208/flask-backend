import re
import nltk
from nltk.corpus import stopwords

# Download stopwords dataset
nltk.download('stopwords')

# Store stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Split sentence into words
    words = text.split()

    # Remove unnecessary stopwords
    words = [word for word in words if word not in stop_words]

    # Join words back into sentence
    cleaned_text = " ".join(words)

    return cleaned_text