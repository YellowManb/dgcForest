# -*- coding:utf-8 -*-
"""
"""
import numpy as np
from .base_layer import BaseLayer
from ..estimators import get_estimator_kfold
from ..utils.metrics import accuracy_pb, accuracy_win_vote, accuracy_win_avg
from ..utils.win_utils import get_windows
from ..utils.debug_utils import repr_blobs_shape
from ..utils.log_utils import get_logger
from gcforest.cascade.fg_conv_cascade import FGConvCascade
LOGGER = get_logger("gcforest_HSI_rw.layers.fg_con_layer")

#CV_POLICYS = ["data", "win"]
#CV_POLICYS = ["data"]

def load_json(path):
    import json
    """
    支持以//开头的注释
    """
    lines = []
    with open(path) as f:
        for row in f.readlines():
            if row.strip().startswith("//"):
                continue
            lines.append(row)
    return json.loads("\n".join(lines))


class FGConvLayer(BaseLayer):
    def __init__(self, layer_config, data_cache, many_win_layers=False, previous_layer_name=None):
        """
        est_config (dict): 
            estimator的config
        win_x, win_y, stride_x, stride_y, pad_x, pad_y (int): 
            configs for windows 
        n_folds(int): default=1
             1 means do not use k-fold
        n_classes (int):
             
        """
        super(FGConvLayer, self).__init__(layer_config, data_cache)
        # estimator
        self.est_configs = self.get_value("estimators", None, list, required=True)
        self.win_x = self.get_value("win_x", None, int, required=True)
        self.win_y = self.get_value("win_y", None, int, required=True)
        self.stride_x = self.get_value("stride_x", 1, int)
        self.stride_y = self.get_value("stride_y", 1, int)
        self.pad_x = self.get_value("pad_x", 0, int)
        self.pad_y = self.get_value("pad_y", 0, int)
        self.n_classes = self.get_value("n_classes", None, int, required=True)
        self.ca_json_dir = self.get_value("ca_json_dir", None, basestring, required=True)

        assert len(self.bottom_names) >= 2
        assert len(self.est_configs) == len(self.top_names), "Each estimator shoud produce one unique top"
        self.eval_metrics = [("predict", accuracy_pb), ("vote", accuracy_win_vote), ("avg", accuracy_win_avg)]
        self.estimator1d = [None for ei in range(len(self.est_configs))]

    def _init_estimators(self, ei, random_state):
        """
        ei (int): estimator index
        """
        top_name = self.top_names[ei]
        est_args = self.est_configs[ei].copy()
        est_name ="{}/{}_folds".format(top_name, est_args["n_folds"])
        # n_folds
        n_folds = int(est_args["n_folds"])
        est_args.pop("n_folds")
        # est_type
        est_type = est_args["type"]
        est_args.pop("type")
        # random_state
        random_state = (random_state + hash("[estimator] {}".format(est_name))) % 1000000007
        return get_estimator_kfold(est_name, n_folds, est_type, est_args, random_state=random_state)

    def fit_transform(self, train_config):
        LOGGER.info("[data][{}], bottoms={}, tops={}".format(self.name, self.bottom_names, self.top_names))
        phases = ["train", "test"]
        X_train_win, y_train_win = None, None
        test_sets = None

        for ti, top_name in enumerate(self.top_names):
            LOGGER.info("[progress][{}] ti={}/{}, top_name={}".format(self.name, ti, len(self.top_names), top_name))
            # check top cache
            if np.all(self.check_top_cache(phases, ti)):
                LOGGER.info("[data][{}] all top cache exists. skip progress".format(self.name))
                continue

            # init X, y, n_classes
            if X_train_win is None:
                for pi, phase in enumerate(phases):
                    bottoms = self.data_cache.gets(phase, self.bottom_names)
                    LOGGER.info('[data][{},{}] bottoms.shape={}'.format(self.name, phase, repr_blobs_shape(bottoms)))
                    X, y = np.concatenate(bottoms[:-1], axis=1), bottoms[-1]
                    # n x n_windows x channel
                    X_win = get_windows(X, self.win_x, self.win_y, self.stride_x, self.stride_y, self.pad_x, self.pad_y,
                                        parallel=False)
                    _, nh, nw, _ = X_win.shape
                    X_win = X_win.reshape((X_win.shape[0], -1, X_win.shape[-1]))
                    y_win = y[:,np.newaxis].repeat(X_win.shape[1], axis=1)
                    if pi == 0:
                        assert self.n_classes == len(np.unique(y)), \
                                "n_classes={}, len(unique(y))={}".format(self.n_classes, len(np.unique(y)))
                        X_train_win, y_train_win = X_win, y_win
                    else:
                        test_sets = [("test", X_win, y_win), ]
            ca_json_dir = self.ca_json_dir
            config = load_json(ca_json_dir)
            cascade = FGConvCascade(config["cascade"])
            n_samples, n_wins, n_dims = X_train_win.shape
            y_probas = cascade.fit_transform(X_train_win.reshape(-1, n_dims), y_train_win.reshape(-1),
                                             test_sets[0][1].reshape(-1, n_dims), test_sets[0][2].reshape(-1))
            for pi, phase in enumerate(phases):
                y_proba = y_probas[pi].reshape((-1, nh, nw, self.n_classes)).transpose((0, 3, 1, 2))
                LOGGER.info('[data][{},{}] tops[{}].shape={}'.format(self.name, phase, ti, y_proba.shape))
                self.data_cache.update(phase, self.top_names[ti], y_proba)
            if train_config.keep_model_in_mem:
                self.estimator1d[ti] = cascade

    def score(self):
        eval_metrics = [("predict", accuracy_pb), ("vote", accuracy_win_vote), ("avg", accuracy_win_avg)]
        for ti, top_name in enumerate(self.top_names):
            for phase in ["train", "test"]:
                y = self.data_cache.get(phase, self.bottom_names[-1])
                y_proba = self.data_cache.get(phase, top_name)
                y_proba = y_proba.transpose((0,2,3,1))
                y_proba = y_proba.reshape((y_proba.shape[0], -1, y_proba.shape[3]))
                y = y[:,np.newaxis].repeat(y_proba.shape[1], axis=1)
                for eval_name, eval_metric in eval_metrics:
                    acc = eval_metric(y, y_proba)
                    LOGGER.info("Accuracy({}.{}.{})={:.2f}%".format(top_name, phase, eval_name, acc*100))
