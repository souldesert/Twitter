import build_model as bm
import create_sample as cs
import numpy as np
import svm_model as svm
import nb_model as nb
import knn_model as knn
import tree_model as tree
import load_etalon_data as led
import load_test_data as ltd
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# pd.prepare_data()
model = bm.build_model()
predictor = np.asarray(cs.create_sample(model))
test = ltd.load_test_data(model)
etalon = led.load_etalon_data()

ITERATIONS = 5
TRAIN_CHUNK_SIZE = round(len(predictor) / ITERATIONS)
TEST_CHUNK_SIZE = round(len(etalon) / ITERATIONS)

knn_f1, knn_recall, knn_precision = [], [], []
nb_f1, nb_recall, nb_precision = [], [], []
svm_f1, svm_recall, svm_precision = [], [], []
tree_f1, tree_recall, tree_precision = [], [], []


for i in range(0, ITERATIONS):
    print("Прогон № " + str(i + 1))

    train_start = i * TRAIN_CHUNK_SIZE
    train_stop = (i+1) * TRAIN_CHUNK_SIZE - 1
    test_start = i * TEST_CHUNK_SIZE
    test_stop = (i+1) * TEST_CHUNK_SIZE - 1
    
    pred_crossval = predictor[train_start:train_stop]
    test_crossval = test[test_start:test_stop]
    etalon_crossval = etalon[test_start:test_stop]

    test_knn = knn.knn_model(pred_crossval, test_crossval)
    knn_f1.append(f1_score(etalon_crossval, test_knn, average='weighted'))
    knn_recall.append(recall_score(etalon_crossval, test_knn, average='weighted'))
    knn_precision.append(precision_score(etalon_crossval, test_knn, average='weighted'))

    test_nb = nb.nb_model(pred_crossval, test_crossval)
    nb_f1.append(f1_score(etalon_crossval, test_nb, average='weighted'))
    nb_recall.append(recall_score(etalon_crossval, test_nb, average='weighted'))
    nb_precision.append(precision_score(etalon_crossval, test_nb, average='weighted'))

    test_svm = svm.svm_model(pred_crossval, test_crossval)
    svm_f1.append(f1_score(etalon_crossval, test_svm, average='weighted'))
    svm_recall.append(recall_score(etalon_crossval, test_svm, average='weighted'))
    svm_precision.append(precision_score(etalon_crossval, test_svm, average='weighted'))

    test_tree = tree.tree_model(pred_crossval, test_crossval)
    tree_f1.append(f1_score(etalon_crossval, test_tree, average='weighted'))
    tree_recall.append(recall_score(etalon_crossval, test_tree, average='weighted'))
    tree_precision.append(precision_score(etalon_crossval, test_tree, average='weighted'))

knn_f1 = np.mean(knn_f1, axis=0).tolist()
knn_recall = np.mean(knn_recall, axis=0).tolist()
knn_precision = np.mean(knn_precision, axis=0).tolist()

nb_f1 = np.mean(nb_f1, axis=0).tolist()
nb_recall = np.mean(nb_recall, axis=0).tolist()
nb_precision = np.mean(nb_precision, axis=0).tolist()

svm_f1 = np.mean(svm_f1, axis=0).tolist()
svm_recall = np.mean(svm_recall, axis=0).tolist()
svm_precision = np.mean(svm_precision, axis=0).tolist()

tree_f1 = np.mean(tree_f1, axis=0).tolist()
tree_recall = np.mean(tree_recall, axis=0).tolist()
tree_precision = np.mean(tree_precision, axis=0).tolist()

print(" Результаты кросс-валидации ".center(100, "=") + "\n")
print(" "*25 + "F1 score".ljust(25) + "Recall".ljust(25) + "Precision".ljust(25) + "\n")
print("K Neighbors".ljust(25) + str(knn_f1).ljust(25) + str(knn_recall).ljust(25) + str(knn_precision).ljust(25))
print("Naive Bayes".ljust(25) + str(nb_f1).ljust(25) + str(nb_recall).ljust(25) + str(nb_precision).ljust(25))
print("Support Vector Machines".ljust(25) + str(svm_f1).ljust(25) + str(svm_recall).ljust(25) + str(svm_precision).ljust(25))
print("Decision Trees".ljust(25) + str(tree_f1).ljust(25) + str(tree_recall).ljust(25) + str(tree_precision).ljust(25))
