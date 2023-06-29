import os
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
if __name__ == '__main__':

    filepath1 = os.path.join('results/UCI/feature_prefix_bin_stride_20000_new', 'test_predictions.npy')
    filepath2 = os.path.join('results/UCI/feature_prefix_bin_stride_20000_new', 'test_labels.npy')
    test_predictions = np.load(filepath1)
    test_labels = np.load(filepath2)

    x = test_labels
    y = [1 if pred > 0.5 else 0 for pred in test_predictions]

    res_accu = metrics.accuracy_score(x, y)
    roc_auc = metrics.roc_auc_score(test_labels, test_predictions)
    res_precision = metrics.precision_score(x, y, average='weighted')
    res_recall = metrics.recall_score(x, y, average='weighted')
    res_f1 = metrics.f1_score(x, y, average='weighted')

    print('Test Accuracy: %.3f' % res_accu)
    print('Test ROC AUC %.3f' % roc_auc)
    print("Weighted!")
    print('Test F1-score: %.3f' % res_f1)
    print('Test Recall: %.3f' % res_recall)
    print('Test Precision: %.3f' % res_precision)

    def get_cm(labels, predictions, p=0.5):
        cm = confusion_matrix(labels, predictions > p)

        print('True Negatives: ', cm[0][0])
        print('False Positives: ', cm[0][1])
        print('False Negatives: ', cm[1][0])
        print('True Positives: ', cm[1][1])
        print('Total Positives: ', np.sum(cm[1]))


    get_cm(test_labels, test_predictions)
