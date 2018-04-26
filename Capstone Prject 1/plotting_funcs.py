"""
All sor of usefull plotting functions
"""
import matplotlib.pyplot as plt
import sklearn.metrics as met
def plot_roc(estimator, X_test, y_test):
    y_score = estimator.decision_function(X_test)
    fpr, tpr, threshold = met.roc_curve(y_test, y_score)
    fig = plt.figure()
    lw = 2
    auc = met.auc(fpr, tpr)
    _= plt.plot(
        fpr,
        tpr,
        color='darkorange',
        lw=lw,
        label='ROC curve (area = {:0.2f})'.format(auc))
    _= plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    _= plt.xlim([0.0, 1.0])
    _= plt.ylim([0.0, 1.01])
    _= plt.xlabel('False Positive Rate')
    _= plt.ylabel('True Positive Rate')
    _= plt.title('ROC')
    _= plt.legend(loc="lower right")
    return fig