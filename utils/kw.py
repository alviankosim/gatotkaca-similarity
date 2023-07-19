# Hey, it's me @alviankosim

from summa import keywords
from keybert import KeyBERT

def kw_textrank(text):
    result = keywords.keywords(text).split('\n')
    
    return result[:5]

def kw_keybert(text):
    kw_model = KeyBERT()
    keyphrases = kw_model.extract_keywords(text)

    cleaned_keyphrases = []
    for kw in keyphrases:
        cleaned_keyphrases.append(kw[0])
    
    return cleaned_keyphrases