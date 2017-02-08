import xml.etree.ElementTree as XmlParser
import process_text as pt


def prepare_data():
    tree = XmlParser.parse('Corpora/ttk_train.xml')
    root = tree.getroot()
    corpora = ''
    for twit in root.iter('column'):
        if twit.attrib['name'] == 'text':
            corpora += twit.text + ' '

    tree = XmlParser.parse('Corpora/ttk_test.xml')
    root = tree.getroot()
    for twit in root.iter('column'):
        if twit.attrib['name'] == 'text':
            corpora += twit.text + ' '

    processed = pt.process_text(corpora)
    # write processed text to file
    processed_text_corpora = open('ttk_train_text.txt', 'wb')
    processed_text_corpora.write(processed.encode('UTF-8'))

# prepare_data()
