# Hey, it's me @alviankosim

import re
import string
import contractions
import unicodedata
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

def preprocess(text, use_stem=True):
    # casefolding
    cleaned_text = text.lower()
    # accented char removal
    cleaned_text = unicodedata.normalize("NFKD", cleaned_text)
    # expanding contractions
    cleaned_text = expand_words(cleaned_text)
    # re number, punctuation, whitespace removal
    cleaned_text = re.sub(r'[^a-zA-z0-9.,!?/:;\"\'\s]', '', cleaned_text)
    cleaned_text = re.sub('[0-9]+', '', cleaned_text)
    cleaned_text = remove_punct(cleaned_text)
    cleaned_text = cleaned_text.strip()
    # removing stopwords
    cleaned_text = remove_stopwords(cleaned_text)
    # stemming
    if (use_stem):
        cleaned_text = stem_words(cleaned_text)

    return cleaned_text

def expand_words(text):
    expanded_words = []
    for word in text.split():
        expanded_words.append(contractions.fix(word))
    
    return ' '.join(expanded_words)

def stem_words(text):
    snow = SnowballStemmer(language='english')

    stemmed_words = []
    words = word_tokenize(text)
    
    for w in words:
        stemmed_words.append(snow.stem(w))
    
    return ' '.join(stemmed_words)

def remove_stopwords(text):
    removed_stopwords = []
    stop_words = stopwords.words('english')
    for word in text.split():
        if word not in stop_words:
            removed_stopwords.append(word)
    
    return ' '.join(removed_stopwords)

def remove_punct(text):
    text_p = "".join([char for char in text if char not in string.punctuation])
    return text_p