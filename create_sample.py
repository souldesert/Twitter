import xml.etree.ElementTree as XmlParser
import nltk
import numpy as np
import process_text as pt


def create_sample(model):

    tree = XmlParser.parse('Corpora/ttk_train.xml')
    root = tree.getroot()
    database = root.find('database')
    train = []
    for twit in database.findall('table'):
        vectors = []
        columns = twit.findall('column')
        text = columns[3].text
        processed = pt.process_text(text)
        processed_words = nltk.word_tokenize(processed)

        # remove ponctuation
        punct = ['.', ',', '!', '?', '%', '..']

        for symbol in punct:
            while symbol in processed_words:
                processed_words.remove(symbol)

        for separate_word in processed_words:
            try:
                vectors.append(model[separate_word])
            except KeyError:
                continue

        vectors = np.asarray(vectors)
        if np.count_nonzero(vectors) != 0:
            average = np.mean(vectors, axis=0).tolist()
        else:
            continue

        for i in range(4, len(columns)):
            if columns[i].text != 'NULL':
                record = []
                if columns[i].text == '+-':
                    opinion = 0
                elif columns[i].text == '--':
                    opinion = -1
                else:
                    opinion = int(columns[i].text)
                record.append(average)
                record.append(opinion)
                record.append(columns[i].attrib['name'])
                train.append(record)
    return train
