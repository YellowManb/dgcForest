# -*- coding:utf-8 -*-
"""
"""
from sklearn.externals import joblib
from .base_estimator import BaseClassifierWrapper
from ..utils.log_utils import get_logger

LOGGER = get_logger('gcforest_HSI_rw.estimators.sklearn_estimators')

def forest_predict_batch_size(clf, X):
    # batch_size = 1000
    import psutil
    free_memory = psutil.virtual_memory().free
    if free_memory < 2e9:
        free_memory = int(2e9)
    max_mem_size = max(int(free_memory * 0.5), int(8e10))
    mem_size_1 = clf.n_classes_ * clf.n_estimators * 16
    batch_size = (max_mem_size - 1) / mem_size_1 + 1
    if batch_size < 10:
        batch_size = 10
    if batch_size >= X.shape[0]:
        return 0
    return batch_size

class SKlearnBaseClassifier(BaseClassifierWrapper):
    def _load_model_from_disk(self, cache_path):
        return joblib.load(cache_path)

    def _save_model_to_disk(self, clf, cache_path):
        joblib.dump(clf, cache_path)

class GCExtraTreesClassifier(SKlearnBaseClassifier):
    def __init__(self, name, kwargs):
        from sklearn.ensemble import ExtraTreesClassifier
        super(GCExtraTreesClassifier, self).__init__(name, ExtraTreesClassifier, kwargs)
    
    def _default_predict_batch_size(self, clf, X):
        return forest_predict_batch_size(clf, X)

class GCRandomForestClassifier(SKlearnBaseClassifier):
    def __init__(self, name, kwargs):
        from sklearn.ensemble import RandomForestClassifier
        super(GCRandomForestClassifier, self).__init__(name, RandomForestClassifier, kwargs)
    
    def _default_predict_batch_size(self, clf, X):
        return forest_predict_batch_size(clf, X)
