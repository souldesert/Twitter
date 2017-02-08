import build_model as bm
import create_sample as cs
import numpy as np
import svm_model as svm
import nb_model as nb
import knn_model as knn
import load_etalon_data as led
from round import round_
from sklearn.metrics import classification_report


# pd.prepare_data()
model = bm.build_model()
predictor = np.asarray(cs.create_sample(model))
etalon = led.load_etalon_data()

test_svm_eval = []
test_nb_eval = []
test_knn_eval = []

ITERATIONS = 10

for i in range(0, ITERATIONS):
    print("Прогон № " + str(i + 1))

    test_nb = nb.nb_model(predictor, model)
    test_svm = svm.svm_model(predictor, model)
    test_knn = knn.knn_model(predictor, model)

    test_nb_single = []
    test_svm_single = []
    test_knn_single = []

    for j in range(0, len(etalon)):
        test_nb_single.append(test_nb[j][3])
        test_svm_single.append(test_svm[j][3])
        test_knn_single.append(test_knn[j][3])

    test_nb_eval.append(test_nb_single)
    test_svm_eval.append(test_svm_single)
    test_knn_eval.append(test_knn_single)

target_names = ['negative', 'neutral', 'positive']

print("\nРезультаты алгоритма Naive Bayes:\n")
print(classification_report(etalon, round_(test_nb_eval), target_names=target_names) + '\n')

print("Результаты алгоритма SVM:\n")
print(classification_report(etalon, round_(test_svm_eval), target_names=target_names) + '\n')

print("Результаты алгоритма K Nearest Neighbors:\n")
print(classification_report(etalon, round_(test_knn_eval), target_names=target_names) + '\n')
