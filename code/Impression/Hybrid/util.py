import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

def confidence_interval(y_ture, y_pred):

    def CI(scores, confidence):
        print('50th percentile (median) = %.3f' % np.median(scores))
        # calculate 95% confidence intervals (100 - alpha)
        alpha = 100 - confidence
        # calculate lower percentile (e.g. 2.5)
        lower_p = alpha / 2.0
        # retrieve observation at lower percentile
        lower = max(0.0, np.percentile(scores, lower_p))
        print('%.1fth percentile = %.3f' % (lower_p, lower))
        # calculate upper percentile (e.g. 97.5)
        upper_p = (100 - alpha) + (alpha / 2.0)
        # retrieve observation at upper percentile
        upper = min(1.0, np.percentile(scores, upper_p))
        print('%.1fth percentile = %.3f' % (upper_p, upper))
    # bootstrap
    precisions = []; recalls = []; fscores = []; accs = []
    for _ in range(1000):
        # bootstrap sample
        indices = np.random.randint(0, len(y_ture), len(y_ture))
        y_true_sample = y_ture[indices]
        y_pred_sample = y_pred[indices]
        # calculate and store statistic
        precision, recall, fscore, non = precision_recall_fscore_support(y_true_sample, y_pred_sample, average='binary'); acc = accuracy_score(y_true_sample, y_pred_sample);
        precisions.append(precision); recalls.append(recall); fscores.append(fscore); accs.append(acc)
    print('Precision:'); CI(precisions, 95);
    print('Recalls:'); CI(recalls, 95);
    print('Accs:'); CI(accs, 95);
    print('Fscores:'); CI(fscores, 95);