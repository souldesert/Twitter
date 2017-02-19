from sklearn.naive_bayes import GaussianNB
import numpy as np


def nb_model(predictor, test_data):
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

    gnb_beeline = GaussianNB()
    gnb_beeline.fit(predictor_beeline, target_beeline)

    gnb_mts = GaussianNB()
    gnb_mts.fit(predictor_mts, target_mts)

    gnb_megafon = GaussianNB()
    gnb_megafon.fit(predictor_megafon, target_megafon)

    gnb_tele2 = GaussianNB()
    gnb_tele2.fit(predictor_tele2, target_tele2)

    gnb_rostelecom = GaussianNB()
    gnb_rostelecom.fit(predictor_rostelecom, target_rostelecom)

    # gnb_skylink = GaussianNB()
    # gnb_skylink.fit(predictor_skylink, target_skylink)

    # test = ltd.load_test_data(model)
    test = test_data
    test_result = []

    for row in test:
        if row[2] == 'beeline':
            result = gnb_beeline.predict(np.reshape(row[1], (1, -1)))
        elif row[2] == 'mts':
            result = gnb_mts.predict(np.reshape(row[1], (1, -1)))
        elif row[2] == 'megafon':
            result = gnb_megafon.predict(np.reshape(row[1], (1, -1)))
        elif row[2] == 'tele2':
            result = gnb_tele2.predict(np.reshape(row[1], (1, -1)))
        elif row[2] == 'rostelecom':
            result = gnb_rostelecom.predict(np.reshape(row[1], (1, -1)))
        elif row[2] == 'komstar':
            result = 0  # gnb_komstar.predict(row[0])
        else:
            result = 0  # gnb_skylink.predict(np.reshape(row[1], (1, -1)))

        try:
            # row.append(result[0])
            test_result.append(result[0])
        except TypeError:
            # row.append(result)
            test_result.append(result)

    return test_result
