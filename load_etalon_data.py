import xml.etree.ElementTree as XmlParser


def load_etalon_data():

    tree = XmlParser.parse('Corpora/eval/ttk_test_etalon.xml')
    root = tree.getroot()
    database = root.find('database')
    etalon = []
    for twit in database.findall('table'):
        mentioned_operators = []
        columns = twit.findall('column')
        for i in range(4, len(columns)):
            if columns[i].text != 'NULL':
                mentioned_operators.append(i)

        for operator_id in mentioned_operators:
            record = int(columns[operator_id].text)
            etalon.append(record)

    return etalon
