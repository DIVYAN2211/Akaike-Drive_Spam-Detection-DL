
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def get_class_weights(labels):
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return weights
