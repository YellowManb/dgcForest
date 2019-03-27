"""
"""
from .base_estimator import BaseClassifierWrapper
from .sklearn_estimators import GCExtraTreesClassifier, GCRandomForestClassifier

from .kfold_wrapper import KFoldWrapper

def get_estimator_class(est_type):
    if est_type == "ExtraTreesClassifier":
        return GCExtraTreesClassifier
    if est_type == "RandomForestClassifier":
        return GCRandomForestClassifier
    raise ValueError('Unkown Estimator Type, est_type={}'.format(est_type))

def get_estimator(name, est_type, est_args):
    est_class = get_estimator_class(est_type)
    return est_class(name, est_args)

def get_estimator_kfold(name, n_splits, est_type, est_args, random_state=None):
    est_class = get_estimator_class(est_type)
    return KFoldWrapper(name, n_splits, est_class, est_args, random_state=random_state)
