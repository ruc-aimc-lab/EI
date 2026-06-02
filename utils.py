# evaluation metrics
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve


def one_hot(x, num_ratings=None):
    if num_ratings is None:
        num_ratings = int(max(x)) + 1
    return np.eye(num_ratings)[x]


def _fast_hist(label_true, label_pred, n_class=2):
    hist = np.bincount((n_class * label_true.astype(int) + label_pred.astype(int)).flatten(),
                       minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


class Measurement(object):
    @staticmethod
    def conf_mats(targets, preds):
        targets = np.array(targets)
        preds = np.array(preds)

        assert preds.shape == targets.shape
        n_class = targets.shape[1]
        hists = np.zeros((n_class, 2, 2))
        for i in range(n_class):
            hists[i] = _fast_hist(targets[:, i].flatten(), preds[:, i].flatten())
        return hists

    @staticmethod
    def conf_mat_based_measurements(hists, e=1e-10):
        n_class = hists.shape[0]
        precisions = np.zeros(n_class)
        recalls = np.zeros(n_class)
        fs = np.zeros(n_class)
        specificities = np.zeros(n_class)
        accuracies = np.zeros(n_class)
        for i in range(n_class):
            hist = hists[i]
            tp = hist[1, 1]
            tn = hist[0, 0]
            fp = hist[0, 1]
            fn = hist[1, 0]

            precisions[i] = tp / (tp + fp + e)
            recalls[i] = tp / (tp + fn + e)
            fs[i] = 2 * tp / (2 * tp + fp + fn + 2 * e)
            specificities[i] = tn / (tn + fp + e)
            accuracies[i] = (tp + tn) / (tp + tn + fp + fn)

        return precisions, recalls, fs, specificities, accuracies

    @staticmethod
    def score_based_measurements(scores, targets):
        assert scores.shape == targets.shape
        im_num, label_num = scores.shape

        aps = np.zeros(label_num)
        iaps = np.zeros(im_num)
        aucs = np.zeros(label_num)

        for i in range(label_num):
            score = scores[:, i]
            target = targets[:, i]
            if (target == 0).all() or (target == 1).all():
                ap = np.nan
                auc = np.nan
            else:
                ap = average_precision_score(target, score)
                auc = roc_auc_score(target, score)
            aps[i] = ap
            aucs[i] = auc
        for i in range(im_num):
            score = scores[i, :]
            target = targets[i, :]
            if (target == 0).all() or (target == 1).all():
                ap = np.nan
            else:
                ap = average_precision_score(target, score)
            iaps[i] = ap

        return aps, iaps, aucs


class Evaluater(Measurement):
    def evaluate(self, scores, targets, thre=0):
        if scores.shape == targets.shape:
            # multi-label
            preds = scores.copy()
            preds[preds>thre] = 1
            preds[preds<=thre] = 0
            preds = preds.astype(int)

            targets = targets.astype(int)

            conf_mat = self.conf_mats(targets, preds)
            precisions, recalls, fs, specificities, accuracies = self.conf_mat_based_measurements(conf_mat)
            aps, iaps, aucs = self.score_based_measurements(scores, targets)

        else:
            # multi-class
            num_class = scores.shape[1]
            
            preds = scores.copy()
            preds = np.argmax(preds, axis=1)
            
            conf_mat_multi_class = _fast_hist(targets, preds, num_class)
            accuracies = np.sum(np.diag(conf_mat_multi_class)) / np.sum(conf_mat_multi_class)
            
            preds_one_hot = one_hot(preds, num_ratings=num_class)
            targets_one_hot = one_hot(targets, num_ratings=num_class)
            
            conf_mat = self.conf_mats(targets_one_hot, preds_one_hot)
            precisions, recalls, fs, specificities, _ = self.conf_mat_based_measurements(conf_mat)
            aps, iaps, aucs = self.score_based_measurements(scores, targets_one_hot)
        
        return conf_mat, recalls, specificities, fs, aucs, aps, accuracies
     