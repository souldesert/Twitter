from nltk.corpus import stopwords
from nltk import stem
import re


def process_text(corpora_to_process):

    # initialize stemmer
    # stemmer = stem.snowball.RussianStemmer()

    sequences_to_remove = ['…', '|', ':)', ':(', ')', '(', ':-(', ':-)', '=)', '=(', '-', '/', '\\', '=&gt;', '–',
                           '//', '—', '-&gt;', '&gt;&lt;', '&gt;&gt;', '&gt;_&lt;']

    sequences_to_strip = ['«', '»', '"', '”', '/', '&amp;laquo;', '&amp;raquo;', '...', ')', '(', ':', '•', '&amp;quot', '^',
                          '&amp;amp;quot;', '…', '&gt;&gt;&gt;', '&amp;nbsp;', '—', '=&gt;', '&gt;&gt;', '&amp;#8230;', ';']

    # remove rubbish 'words'
    processed_corpora = ' '.join([word.lower() for word in corpora_to_process.split() if word not in sequences_to_remove])

    for sequence in sequences_to_strip:
        processed_corpora = ' '.join([word.replace(sequence, '') for word in processed_corpora.split()])

    # remove stopwords
    processed_corpora = ' '.join([word.lstrip("#") for word in processed_corpora.split() if word not in (stopwords.words('russian'))])

    # remove URLs
    processed_corpora = ' '.join([word for word in processed_corpora.split() if 'http' not in word])

    # remove @mentions
    processed_corpora = ' '.join([word.lstrip("@") for word in processed_corpora.split()])

    # processed_corpora_stemmed = ' '
    # # stem
    # for word in processed_corpora.split():
    #     if re.search('[a-zA-Z-9]', word):
    #         processed_corpora_stemmed.join(word)
    #     else:
    #         processed_corpora_stemmed.join(stemmer.stem(word))
    #     processed_corpora_stemmed.join(' ')

    #processed_corpora = ' '.join([stemmer.stem(word) for word in processed_corpora.split() if not re.search('[a-zA-Z-9]', word)])

    return processed_corpora
