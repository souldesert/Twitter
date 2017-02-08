from sklearn import svm
import numpy as np
import load_test_data as ltd


def svm_model(predictor, model):
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

    svm_model_beeline = svm.SVC(kernel='rbf', C=50, gamma=100)
    svm_model_beeline.fit(predictor_beeline, target_beeline)

    svm_model_mts = svm.SVC(kernel='rbf', C=50, gamma=100)
    svm_model_mts.fit(predictor_mts, target_mts)

    svm_model_megafon = svm.SVC(kernel='rbf', C=50, gamma=100)
    svm_model_megafon.fit(predictor_megafon, target_megafon)

    svm_model_tele2 = svm.SVC(kernel='rbf', C=50, gamma=100)
    svm_model_tele2.fit(predictor_tele2, target_tele2)

    svm_model_rostelecom = svm.SVC(kernel='rbf', C=50, gamma=100)
    svm_model_rostelecom.fit(predictor_rostelecom, target_rostelecom)

    # svm_model_komstar = svm.SVC(kernel='rbf', C=50, gamma=100)
    # svm_model_komstar.fit(predictor_komstar, target_komstar)

    svm_model_skylink = svm.SVC(kernel='rbf', C=50, gamma=100)
    svm_model_skylink.fit(predictor_skylink, target_skylink)

    test = ltd.load_test_data(model)

    for row in test:
        if row[2] == 'beeline':
            result = svm_model_beeline.predict(np.reshape(row[1], (1, -1)))
        elif row[2] == 'mts':
            result = svm_model_mts.predict(np.reshape(row[1], (1, -1)))
        elif row[2] == 'megafon':
            result = svm_model_megafon.predict(np.reshape(row[1], (1, -1)))
        elif row[2] == 'tele2':
            result = svm_model_tele2.predict(np.reshape(row[1], (1, -1)))
        elif row[2] == 'rostelecom':
            result = svm_model_rostelecom.predict(np.reshape(row[1], (1, -1)))
        elif row[2] == 'komstar':
            result = 0  # svm_model_komstar.predict(np.reshape(row[1], (1, -1)))
        else:
            result = svm_model_skylink.predict(np.reshape(row[1], (1, -1)))

        try:
            row.append(result[0])
        except TypeError:
            row.append(result)

    return test
