# -*- coding:utf-8 -*-
"""
"""
import numpy as np
from ..utils.log_utils import get_logger

LOGGER = get_logger('gcforest_HSI_rw.estimators.est_utils')

def xgb_train(train_config, X_train, y_train, X_test, y_test):
    import xgboost as xgb
    LOGGER.info("X_train.shape={}, y_train.shape={}, X_test.shape={}, y_test.shape={}".format(
        X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    param = train_config["param"]
    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_test = xgb.DMatrix(X_test, label=y_test)
    num_round = int(train_config["num_round"])
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    try:
        bst = xgb.train(param, xg_train, num_round, watchlist)
    except KeyboardInterrupt:
        LOGGER.info("Canceld by user's Ctrl-C action")
        return
    y_pred = np.argmax(bst.predict(xg_test), axis=1)
    acc = 100. * np.sum(y_pred == y_test) / len(y_test)
    LOGGER.info("accuracy={}%".format(acc))
