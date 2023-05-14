

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

import os


def save_config(run_name, model_name, model_is_pretrained, num_classes, optimizer_name, lr, augmentation, regularization):
    config_str = '''# Run name
run_name = "%s"

# Network architecture
# mlp, resnet18, resnet50, vgg11
model_name = "%s"
model_is_pretrained = %s
num_classes = %s

# Optimizer
# SGD, Adagrad, Adam
optimizer_name = "%s"
lr = %f

# Data augmentation
augmentation = "%s"

# Regularization
regularization = "%s"
    '''%(run_name, model_name, model_is_pretrained, num_classes, optimizer_name, lr, augmentation, regularization)
    print(config_str)
    os.makedirs("./outputs/runs/%s"%run_name, exist_ok=True)
    with open("./outputs/runs/%s/config.txt"%run_name, "w") as f:
        f.write(config_str)