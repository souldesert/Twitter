from sklearn import neighbors
import numpy as np
import load_test_data as ltd


K_NEIGHBORS = 25


def knn_model(predictor, test_data):

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

    knn_model_beeline = neighbors.KNeighborsClassifier(K_NEIGHBORS, 'distance')
    knn_model_beeline.fit(predictor_beeline, target_beeline)
    
    knn_model_mts = neighbors.KNeighborsClassifier(K_NEIGHBORS, 'distance')
    knn_model_mts.fit(predictor_mts, target_mts)
    
    knn_model_megafon = neighbors.KNeighborsClassifier(K_NEIGHBORS, 'distance')
    knn_model_megafon.fit(predictor_megafon, target_megafon)
    
    knn_model_tele2 = neighbors.KNeighborsClassifier(K_NEIGHBORS, 'distance')
    knn_model_tele2.fit(predictor_tele2, target_tele2)
    
    knn_model_rostelecom = neighbors.KNeighborsClassifier(K_NEIGHBORS, 'distance')
    knn_model_rostelecom.fit(predictor_rostelecom, target_rostelecom)


    # knn_model_komstar = neighbors.KNeighborsClassifier(K_NEIGHBORS, 'distance')
    # knn_model_komstar.fit(predictor_komstar, target_komstar)
    #
    # knn_model_skylink = neighbors.KNeighborsClassifier(K_NEIGHBORS, 'distance')
    # knn_model_skylink.fit(predictor_skylink, target_skylink)

    # test = ltd.load_test_data(model)
    # тест - список
    test = test_data
    test_result = []

    for row in test:
        if row[2] == 'beeline':
            result = knn_model_beeline.predict(np.reshape(row[1], (1, -1)))
        elif row[2] == 'mts':
            result = knn_model_mts.predict(np.reshape(row[1], (1, -1)))
        elif row[2] == 'megafon':
            result = knn_model_megafon.predict(np.reshape(row[1], (1, -1)))
        elif row[2] == 'tele2':
            result = knn_model_tele2.predict(np.reshape(row[1], (1, -1)))
        elif row[2] == 'rostelecom':
            result = knn_model_rostelecom.predict(np.reshape(row[1], (1, -1)))
        elif row[2] == 'komstar':
            result = 0  # knn_model_komstar.predict(row[0])
        else:
            result = 0 # knn_model_skylink.predict(np.reshape(row[1], (1, -1)))

        try:
            #row.append(result[0])
            test_result.append(result[0])
        except TypeError:
            #row.append(result)
            test_result.append(result)

    return test_result
