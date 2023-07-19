import math

# Term frequency each words
def tf(text, use_normalize = True):
    tf_ = {}
    for word in text.split():
        if word in tf_:
            tf_[word] += 1
        else:
            tf_[word] = 1

    if (use_normalize == False):
        return tf_

    tf_normalized = tf_normalize(tf_, text)
    return tf_normalized

# Term normalize the value within the whole text
def tf_normalize(tf_obj, text):
    normalized = {}
    for word in tf_obj:
        normalized[word] = tf_obj[word] / float(len(text.split()))

    return normalized

# IDF(x) = 1 + loge(Total Number Of Documents / Number Of Documents with term x in it)
def idf(term, corpus):
    counting = 0

    for doc in corpus:
        if term in doc:
            counting += 1

    if (counting == 0):
        return 1.0

    return 1.0 + math.log(len(corpus) / counting)

def idf_doc(corpus):
    idf_ = {}
    for doc in corpus:
        for word in doc.split():
            idf_[word] = idf(word, corpus)
    
    return idf_

# tf * idf
def tf_idf(text, tf_normalized, idfed):
    tf_x_idf = {}
    for word in text.split():
        tf_val = 0.0
        if word in tf_normalized:
            tf_val = tf_normalized[word]

        idf_val = 0.0
        if word in idfed:
            idf_val = idfed[word]

        tf_x_idf[word] = tf_val * idf_val
    
    return tf_x_idf

# tfidf -- Cosine Similarity(Query,Document1) = Dot product(Query, Document1) / ||Query|| * ||Document1||
def cos_sim(query, document):
    dot_product = 0
    calc_query = 0
    calc_document = 0
    for word in query:
        dot_product += query[word] * document[word]
        calc_query += query[word] ** 2
        calc_document += document[word] ** 2

    return dot_product / math.sqrt(calc_query) * math.sqrt(calc_document)