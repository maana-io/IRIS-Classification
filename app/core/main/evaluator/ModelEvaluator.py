'''
Leave one out model evaluator
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report




class ModelEvaluator(object):

    ME_LeaveOneOut = "LEAVE_ONE_OUT"
    ME_KFoldXVal = "K_FOLDS"

    supported_modes = [ME_LeaveOneOut, ME_KFoldXVal]

    def __init__(self, lr, fm, lm, feature_data, label_data, topN = 1):
        self.__lr = lr
        self.__fm = fm
        self.__lm = lm
        self.__X = self.__fm.fit_transform(feature_data)
        self.__Y = self.__lm.fit_transform(label_data)
        self.__topN = topN



    def eval(self, mode, nfolds):
        labels, _, _, conf_mat, cls_report = self.eval_data(mode, nfolds)
        cls_report_text = ("%s" % cls_report).split("\n")

        nCols = max(5, len(labels) + 1)
        table_data = []
        table_data.append(["."] + [""] * (nCols - 1))
        table_data.append(["Confusion Matrix".upper()] + [""] * (nCols - 1))
        table_data.append(["."] + [""] * (nCols - 1))
        table_data.append(["Actual\\Predicted"] + list(labels) + [""] * (nCols - len(labels) - 1))
        for (row_no, row) in enumerate(conf_mat):
            table_data.append([labels[row_no]] + [str(val) for val in row] + [""] * (nCols - len(labels) - 1))

        table_data.append(["."] + [""] * (nCols - 1))
        table_data.append(["Classification Report".upper()] + [""] * (nCols - 1))
        table_data.append(["."] + [""] * (nCols - 1))
        for (lin_no, txt_line) in enumerate(cls_report_text):
            if lin_no == 0:
                table_data.append(["Class", "Precision", "Recall", "F1-score", "Support"] + [""] * (nCols - 5))
            else:
                lin_dat = txt_line.split()
                if len(lin_dat) < 1:
                    table_data.append([""] * nCols)
                else:
                    table_data.append([' '.join(lin_dat[:-4])] + lin_dat[-4:] + [""] * (nCols - 5))

        return pd.DataFrame(table_data, columns=list(map(lambda n: "column " + str(n+1), range(nCols))))



    def eval_data(self, mode, nfolds, output_dict = False):
        assert mode in ModelEvaluator.supported_modes, "Invalid splitting mode %s. Supported modes are %s" % \
             (mode, ModelEvaluator.supported_modes)
        if mode == ModelEvaluator.ME_KFoldXVal:
            assert nfolds > 1 and nfolds <= len(self.__Y), "Invalid num-folds %d" % nfolds

        spliter = None
        if mode == ModelEvaluator.ME_LeaveOneOut:
            spliter = LeaveOneOut()
        elif mode == ModelEvaluator.ME_KFoldXVal:
            spliter = StratifiedKFold(n_splits=nfolds, shuffle=True)

        pred_classes = np.zeros(len(self.__Y))
        for train_idx, test_idx in spliter.split(self.__X, self.__Y):
            X_train = self.__X[train_idx]
            X_test =  self.__X[test_idx]
            Y_train = self.__Y[train_idx]
            Y_test = self.__Y[test_idx]

            self.__lr.fit(X_train, Y_train)

            if self.__topN > 1:
                list_of_probs = self.__lr.predict_proba(X_test)
                list_of_prob_with_index = list(map(lambda probs: zip(probs, range(len(probs))), list_of_probs))
                list_of_sorted_prob_with_index = list(map(
                    lambda prob_with_index: sorted(prob_with_index, key = lambda prob_idx: -prob_idx[0]),
                                                     list_of_prob_with_index))
                topN_preds = list(map(lambda sorted_prob_with_index: [idx for (prob, idx) in sorted_prob_with_index[:self.__topN]],
                                 list_of_sorted_prob_with_index))
                topN_preds_and_true_lbl = zip(topN_preds, Y_test)
                pred_classes[test_idx] = list(map(lambda list_preds__true_lbl:
                                             list_preds__true_lbl[1] if list_preds__true_lbl[1] in list_preds__true_lbl[0]
                                             else list_preds__true_lbl[0][0], topN_preds_and_true_lbl))
            else:
                pred_classes[test_idx] = self.__lr.predict(X_test)

        labels = self.__lm.inverse_transform(range(max(self.__Y)+1))
        lbl_true = self.__lm.inverse_transform(self.__Y)
        lbl_pred = self.__lm.inverse_transform(pred_classes.astype(int))
        conf_mat = confusion_matrix(lbl_true, lbl_pred, labels)
        cls_report = classification_report(lbl_true, lbl_pred, target_names=labels, output_dict=output_dict)

        return labels, lbl_true, lbl_pred, conf_mat, cls_report




    def __str__(self):
        return 'Model evaluator.'