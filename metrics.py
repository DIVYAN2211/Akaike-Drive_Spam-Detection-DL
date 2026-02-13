
from sklearn.metrics import classification_report

def print_metrics(y_true, y_pred):
    print(classification_report(y_true, y_pred, target_names=["Ham","Spam"]))
