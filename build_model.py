from gensim.models import Word2Vec
import codecs
import nltk
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def build_model():
    raw = codecs.open('ttk_train_text.txt', 'r', 'utf-8').read()
    raw_sentences = nltk.sent_tokenize(raw)

    punct = ['.', ',', '!', '?', '%', '..']

    sentences = []

    for raw_sent in raw_sentences:
        sent = nltk.word_tokenize(raw_sent)
        sentences.append(sent)

    for sent in sentences:

        for symbol in punct:
            while symbol in sent:
                sent.remove(symbol)

    return Word2Vec(sentences, min_count=1)

