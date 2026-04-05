import re
import nltk

nltk.download('stopwords')
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'


def preprocess_text(text):

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    tokens = word_tokenize(text)

    tokens = [w for w in tokens if w not in stop_words]

    pos_tags = pos_tag(tokens)

    tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in pos_tags
    ]

    return " ".join(tokens)


if __name__=="__main__":
    text = "Hello, how are you?"
    preprocessed_text = preprocess_text(text)
    print(preprocessed_text)