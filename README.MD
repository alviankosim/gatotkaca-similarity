## Gatot Kaca - joural similarity

a python implementation of journal similarity using TF-IDF and Cosine Similarity (including 2 keyword extraction method).

## Demo
![Gatotkaca Journal Similarity in action](./docs/demo.gif)

### The process outline
- Data reading from CSV (from scopus1.csv)
- Data preprocessing
    - Cleaning data (normalize, remove punctuation, whitespace, etc )
    - Removing stopwords
    - Stemming english words using Snowball Stemmer
- TF-IDF
- Cosine Similarity

#### How to start?
- git clone this project
- then run `python main.py`
- access `127.0.0.1:3600`


by @alviankosim