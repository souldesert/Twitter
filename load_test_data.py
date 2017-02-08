import xml.etree.ElementTree as XmlParser
import nltk
import numpy as np
import process_text as pt


def load_test_data(model):

    tree = XmlParser.parse('Corpora/eval/ttk_test_etalon.xml')
    root = tree.getroot()
    database = root.find('database')
    test = []
    for twit in database.findall('table'):
        mentioned_operators = []

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

        beeline_euph = ['билайн', 'билаин', 'beeline', 'вымпелком', 'пчелайн']
        mts_euph = ['мтс', 'mts', 'мобильные телесистемы']
        megafon_euph = ['мегафон', 'megafon']
        tele2_euph = ['tele2', 'теле2', 'теле 2']
        rostelecom_euph = ['ростелеком', 'rostelecom']
        komstar_euph = ['комстар']
        skylink_euph = ['скайлинк']

        for beeline in beeline_euph:
            if beeline in processed:
                mentioned_operators.append('beeline')
                break
        for mts in mts_euph:
            if mts in processed:
                mentioned_operators.append('mts')
                break
        for megafon in megafon_euph:
            if megafon in processed:
                mentioned_operators.append('megafon')
                break
        for tele2 in tele2_euph:
            if tele2 in processed:
                mentioned_operators.append('tele2')
                break
        for rostelecom in rostelecom_euph:
            if rostelecom in processed:
                mentioned_operators.append('rostelecom')
                break
        for komstar in komstar_euph:
            if komstar in processed:
                mentioned_operators.append('komstar')
                break
        for skylink in skylink_euph:
            if skylink in processed:
                mentioned_operators.append('skylink')
                break

        for operator in mentioned_operators:
            record = [int(columns[0].text), average, operator]
            test.append(record)

    return test
