from sklearn.tree import DecisionTreeClassifier
import numpy as np


def tree_model(predictor, test_data):
    predictor_beeline = np.array(list(predictor[predictor[:, 2] == 'beeline'][:, 0]))
    target_beeline = np.array(list(predictor[predictor[:, 2] == 'beeline'][:, 1]))

    predictor_mts = np.array(list(predictor[predictor[:, 2] == 'mts'][:, 0]))
    target_mts = np.array(list(predictor[predictor[:, 2] == 'mts'][:, 1]))

    predictor_megafon = np.array(list(predictor[predictor[:, 2] == 'megafon'][:, 0]))
    target_megafon = np.array(list(predictor[predictor[:, 2] == 'megafon'][:, 1]))

    predictor_tele2 = np.array(list(predictor[predictor[:, 2] == 'tele2'][:, 0]))
    target_tele2 = np.array(list(predictor[predictor[:, 2] == 'tele2'][:, 1]))

    predictor_rostelecom = np.array(list(predictor[predictor[:, 2] == 'rostelecom'][:, 0]))
    target_rostelecom = np.array(list(predictor[predictor[:, 2] == 'rostelecom'][:, 1]))

    predictor_komstar = np.array(list(predictor[predictor[:, 2] == 'komstar'][:, 0]))
    target_komstar = np.array(list(predictor[predictor[:, 2] == 'komstar'][:, 1]))

    predictor_skylink = np.array(list(predictor[predictor[:, 2] == 'skylink'][:, 0]))
    target_skylink = np.array(list(predictor[predictor[:, 2] == 'skylink'][:, 1]))

    tree_model_beeline = DecisionTreeClassifier(max_depth=5)
    tree_model_beeline.fit(predictor_beeline, target_beeline)

    tree_model_mts = DecisionTreeClassifier(max_depth=5)
    tree_model_mts.fit(predictor_mts, target_mts)

    tree_model_megafon = DecisionTreeClassifier(max_depth=5)
    tree_model_megafon.fit(predictor_megafon, target_megafon)

    tree_model_tele2 = DecisionTreeClassifier(max_depth=5)
    tree_model_tele2.fit(predictor_tele2, target_tele2)

    tree_model_rostelecom = DecisionTreeClassifier(max_depth=5)
    tree_model_rostelecom.fit(predictor_rostelecom, target_rostelecom)

    test = test_data
    test_result = []

    for row in test:
        if row[2] == 'beeline':
            result = tree_model_beeline.predict(np.reshape(row[1], (1, -1)))
        elif row[2] == 'mts':
            result = tree_model_mts.predict(np.reshape(row[1], (1, -1)))
        elif row[2] == 'megafon':
            result = tree_model_megafon.predict(np.reshape(row[1], (1, -1)))
        elif row[2] == 'tele2':
            result = tree_model_tele2.predict(np.reshape(row[1], (1, -1)))
        elif row[2] == 'rostelecom':
            result = tree_model_rostelecom.predict(np.reshape(row[1], (1, -1)))
        elif row[2] == 'komstar':
            result = 0  # tree_model_komstar.predict(np.reshape(row[1], (1, -1)))
        else:
            result = 0  # tree_model_skylink.predict(np.reshape(row[1], (1, -1)))

        try:
            # row.append(result[0])
            test_result.append(result[0])
        except TypeError:
            # row.append(result)
            test_result.append(result)

    return test_result
