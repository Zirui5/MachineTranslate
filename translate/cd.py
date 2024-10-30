import nltk
def unusual_words(text):
    text_vocab = set([w.lower() for w in text if w.isalpha()])
    english_vocab = set([w.lower() for w in nltk.corpus.words.words()])
    unusual = text_vocab.difference(english_vocab)#集合的差集
    return sorted(unusual)


print(nltk.corpus.stopwords.words('French'))#法语停用词
