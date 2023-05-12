

labels = {
    "buildings": 0, 
    "forest": 1, 
    "glacier": 2, 
    "mountain": 3, 
    "sea": 4, 
    "street": 5
}

def label_str2int(class_name: str) -> int:
    if class_name in labels:
        return labels[class_name]
    else:
        raise ValueError("Invalid label name: %s" % class_name)

def label_int2str(label_id: int) -> str:
    if label_id>=0 and label_id<len(labels.keys()):
        return list(labels.keys())[label_id]
    else:
        raise ValueError("Invalid label id: %s" % label_id)

import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn import metrics
def clustering_accuracy(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
    _make_cost_m = lambda x:-x + np.max(x)
    indexes = linear_assignment(_make_cost_m(cm))
    indexes = np.concatenate([indexes[0][:,np.newaxis],indexes[1][:,np.newaxis]], axis=-1)
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    acc = np.trace(cm2) / np.sum(cm2)
    return acc