import xml.etree.ElementTree as XmlParser
from nltk.corpus import stopwords
import re


def process_text():

    # remove stopwords
    processed_corpora = ' '.join([word for word in corpora.split() if word not in (stopwords.words('russian'))])

    # remove URLs
    processed_corpora = ' '.join([word for word in processed_corpora.split() if not re.search(r'https?:\/\/.*[\r\n]*', word)])

    # remove @mentions
    processed_corpora = ' '.join([word for word in processed_corpora.split() if not re.search(r'@\w*', word)])

    # should #hashtags be deleted?
    # processed_corpora = u' '.join([word for word in processed_corpora.split() if not re.search(u'#\w*', word)])

    # write processed text to file
    processed_text_corpora = open('ttk_train_text_processed.txt', 'wb')
    processed_text_corpora.write(processed_corpora.encode('UTF-8'))


tree = XmlParser.parse('Corpora/ttk_train.xml')
root = tree.getroot()
corpora = ''
for twit in root.iter('column'):
    if twit.attrib['name'] == 'text':
        corpora += twit.text + ' '  # + '\n'
text_corpora = open('ttk_train_text.txt', 'wb')
text_corpora.write(corpora.encode('UTF-8'))


process_text()
