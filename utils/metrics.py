
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate(y_true,y_pred,prob):

    acc = accuracy_score(y_true,y_pred)
    pre = precision_score(y_true,y_pred)
    sen = recall_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred)
    auc = roc_auc_score(y_true,prob)

    return acc,pre,sen,f1,auc
