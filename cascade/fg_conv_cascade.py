# -*- coding:utf-8 -*-
"""
"""
import sys, os, os.path as osp
import numpy as np
import pickle
from ..estimators import get_estimator_kfold
from ..utils.config_utils import get_config_value
from ..utils.log_utils import get_logger
from ..utils.metrics import accuracy_pb

LOGGER = get_logger('gcforest_HSI_rw.cascade.cascade_classifier')

def check_dir(path):
    d = osp.abspath(osp.join(path, osp.pardir))
    if not osp.exists(d):
        os.makedirs(d)

def calc_accuracy(y_true, y_pred, name, prefix=""):
    acc = 100. * np.sum(np.asarray(y_true)==y_pred) / len(y_true)
    LOGGER.info('{}Accuracy({})={:.2f}%'.format(prefix, name, acc))
    return acc
def calc_aa_accuracy(y_true, y_pred, name, prefix=""):
    acc = []
    unique_y = np.unique(np.asarray(y_true))
    for i in range(len(unique_y)):
        each_y_index = []
        for j in range(len(y_true)):
            if y_true[j] == unique_y[i]:
                each_y_index.append(j)
        acc.append(100.0 * np.sum(y_pred[each_y_index] == unique_y[i])/len(each_y_index))
    acc = np.asanyarray(acc)
    LOGGER.info('{}AA Accuracy({})={:.2f}%'.format(prefix, name, np.mean(acc)))
    return acc
def calc_kappa(y_true, y_pred, name, prefix=""):
    from sklearn.metrics import cohen_kappa_score
    kappa = 100.0*cohen_kappa_score(y_pred, y_true)
    LOGGER.info('{}AA Accuracy({})={:.2f}%'.format(prefix, name, kappa))
    return kappa
def get_opt_layer_id(acc_list):
    """ Return layer id with max accuracy on training data """
    opt_layer_id = np.argsort(-np.asarray(acc_list), kind='mergesort')[0]
    return opt_layer_id


class FGConvCascade(object):
    def __init__(self, ca_config):
        """
        Parameters (ca_config)
        ----------
        early_stopping_rounds: int
            when not None , means when the accuracy does not increase in early_stopping_rounds, the cascade level will stop automatically growing
        max_layers: int
            maximum number of cascade layers allowed for exepriments, 0 means use Early Stoping to automatically find the layer number
        n_classes: int
            Number of classes
        est_configs: 
            List of CVEstimator's config
        look_indexs_cycle (list 2d): default=None
            specification for layer i, look for the array in look_indexs_cycle[i % len(look_indexs_cycle)] 
            defalut = None <=> [range(n_groups)]
            .e.g.
                look_indexs_cycle = [[0,1],[2,3],[0,1,2,3]]
                means layer 1 look for the grained 0,1; layer 2 look for grained 2,3; layer 3 look for every grained, and layer 4 cycles back as layer 1
        data_save_rounds: int [default=0]
        data_save_dir: str [default=None]
            each data_save_rounds save the intermidiate results in data_save_dir 
            if data_save_rounds = 0, then no savings for intermidiate results
        """
        self.ca_config = ca_config
        self.early_stopping_rounds = self.get_value("early_stopping_rounds", None, int, required=True)
        self.max_layers = self.get_value("max_layers", 0, int)
        self.n_classes = self.get_value("n_classes", None, int, required=True)
        self.est_configs = self.get_value("estimators", None, list, required=True)
        self.look_indexs_cycle = self.get_value("look_indexs_cycle", None, list)
        self.random_state = self.get_value("random_state", None, int)
        self.data_save_dir = self.get_value("data_save_dir", None, basestring)
        self.data_save_rounds = self.get_value("data_save_rounds", 0, int)
        if self.data_save_rounds > 0:
            assert self.data_save_dir is not None, "data_save_dir should not be null when data_save_rounds>0"
        self.eval_metrics = [("predict", accuracy_pb)]
    @property
    def n_estimators_1(self):
        # estimators of one layer
        return len(self.est_configs)
    
    def get_value(self, key, default_value, value_types, required=False):
        return get_config_value(self.ca_config, key, default_value, value_types, 
                required=required, config_name="cascade")

    def _init_estimators(self, li, ei):
        est_args = self.est_configs[ei].copy()
        est_name ="layer_{} - estimator_{} - {}_folds".format(li, ei, est_args["n_folds"])
        # n_folds
        n_folds = int(est_args["n_folds"])
        est_args.pop("n_folds")
        # est_type
        est_type = est_args["type"]
        est_args.pop("type")
        # random_state
        if self.random_state is not None:
            random_state = (self.random_state + hash("[estimator] {}".format(est_name))) % 1000000007
        else:
            random_state = None
        return get_estimator_kfold(est_name, n_folds, est_type, est_args, random_state=random_state)

    def fit_transform(self, X_groups_train, y_train, X_groups_test, y_test, stop_by_test=False, VI=False):
        """
        fit until the accuracy converges in early_stop_rounds 
        stop_by_test: (bool)
            When X_test, y_test is validation data that used for determine the opt_layer_id,
            use this option
        """
        if not type(X_groups_train) == list:
            X_groups_train = [X_groups_train]
        if not type(X_groups_test) == list:
            X_groups_test = [X_groups_test]
        LOGGER.info("X_groups_train.shape={},y_train.shape={},X__groups_test.shape={},y_test.shape={}".format(
            [xr.shape for xr in X_groups_train], y_train.shape, [xt.shape for xt in X_groups_test], y_test.shape))
        n_groups = len(X_groups_train)
        # check look_indexs_cycle
        if self.look_indexs_cycle is None:
            look_indexs_cycle = [list(range(n_groups))]
        else:
            look_indexs_cycle = self.look_indexs_cycle
            for look_indexs in look_indexs_cycle:
                if np.max(look_indexs) >= n_groups or np.min(look_indexs) < 0 or len(look_indexs) == 0:
                    raise ValueError("look_indexs unlegal!!! look_indexs={}".format(look_indexs))
        # init groups
        group_starts, group_ends, group_dims = [], [], []
        n_trains = X_groups_train[0].shape[0]
        n_tests = X_groups_test[0].shape[0]
        X_train = np.zeros((n_trains, 0), dtype=X_groups_train[0].dtype)
        X_test = np.zeros((n_tests, 0), dtype=X_groups_test[0].dtype)
        for i, X_group in enumerate(X_groups_train):
            assert(X_group.shape[0] == n_trains)
            X_group = X_group.reshape(n_trains, -1)
            group_dims.append( X_group.shape[1] )
            group_starts.append(i if i == 0 else group_starts[i - 1] + group_dims[i])
            group_ends.append(group_starts[i] + group_dims[i])
            X_train = np.hstack((X_train, X_group))
        LOGGER.info("group_dims={}".format(group_dims))
        for i, X_group in enumerate(X_groups_test):
            assert(X_group.shape[0] == n_tests)
            X_group = X_group.reshape(n_tests, -1)
            assert(X_group.shape[1] == group_dims[i])
            X_test = np.hstack((X_test, X_group))
        LOGGER.info("X_train.shape={},X_test.shape={}".format(X_train.shape, X_test.shape))

        n_classes = self.n_classes
        n_estimators = self.n_estimators_1
        assert n_classes == len(np.unique(y_train)), "n_classes({}) != len(unique(y)) {}".format(n_classes, np.unique(y_train))
        train_acc_list = []
        test_acc_list = []
        train_aa_list = []
        test_aa_list = []
        kappa_list = []
        # bool_feature_importance=False
        from sklearn.ensemble import RandomForestClassifier
        RF = RandomForestClassifier(n_estimators=500, n_jobs=-1)
        RF.fit(X_train, y_train)
        print(RF.score(X_test, y_test))
        VI=False
        y_train_proba_all = []
        y_test_proba_all = []
        opt_datas = [None, None]
        try:
            # probability of each cascades's estimators
            X_proba_train = np.zeros((X_train.shape[0],n_classes*self.n_estimators_1), dtype=np.float32)
            X_proba_test = np.zeros((X_test.shape[0],n_classes*self.n_estimators_1), dtype=np.float32)
            X_cur_train, X_cur_test = None, None
            layer_id = 0
            while 1:
                if self.max_layers > 0 and layer_id >= self.max_layers:
                    break
                # Copy previous cascades's probability into current X_cur
                if layer_id == 0:
                    # first layer not have probability distribution
                    X_cur_train = np.zeros((n_trains, 0), dtype=np.float32)
                    X_cur_test = np.zeros((n_tests, 0), dtype=np.float32)
                else:
                    X_cur_train = X_proba_train.copy()
                    X_cur_test = X_proba_test.copy()
                # Stack data that current layer needs in to X_cur
                look_indexs = look_indexs_cycle[layer_id % len(look_indexs_cycle)]
                for _i, i in enumerate(look_indexs):
                    X_cur_train = np.hstack((X_cur_train, X_train[:,group_starts[i]:group_ends[i]]))
                    X_cur_test = np.hstack((X_cur_test, X_test[:,group_starts[i]:group_ends[i]]))
                LOGGER.info("[layer={}] look_indexs={}, X_cur_train.shape={}, X_cur_test.shape={}".format(
                    layer_id, look_indexs, X_cur_train.shape, X_cur_test.shape))
                # Fit on X_cur, predict to update X_proba
                y_train_proba_li = np.zeros((y_train.shape[0], n_classes))
                y_test_proba_li = np.zeros((y_test.shape[0], n_classes))
                for ei, est_config in enumerate(self.est_configs):
                    est = self._init_estimators(layer_id, ei)
                    # fit_trainsform
                    y_probas = est.fit_transform(X_cur_train, y_train, y_train,
                        test_sets=[("test", X_cur_test, y_test)], eval_metrics=self.eval_metrics,
                        keep_model_in_mem=False)
                    # train
                    X_proba_train[:,ei*n_classes:ei*n_classes+n_classes] = y_probas[0]
                    y_train_proba_li += y_probas[0]
                    # test
                    X_proba_test[:,ei*n_classes:ei*n_classes+n_classes] = y_probas[1]
                    y_test_proba_li += y_probas[1]
                y_train_proba_li /= len(self.est_configs)
                y_test_proba_li /= len(self.est_configs)
                y_train_proba_all.append(y_train_proba_li)
                y_test_proba_all.append(y_test_proba_li)
                train_avg_acc = calc_accuracy(y_train, np.argmax(y_train_proba_li, axis=1), 'layer_{} - train.classifier_average'.format(layer_id))
                test_avg_acc = calc_accuracy(y_test, np.argmax(y_test_proba_li, axis=1), 'layer_{} - test.classifier_average'.format(layer_id))
                train_aa_acc = calc_aa_accuracy(y_train, np.argmax(y_train_proba_li, axis=1),
                                                'layer_{} - train.classifier_AA_average'.format(layer_id))
                test_aa_acc = calc_aa_accuracy(y_test, np.argmax(y_test_proba_li, axis=1),
                                               'layer_{} - test.classifier_AA_average'.format(layer_id))
                kappa_list.append(calc_kappa(y_test, np.argmax(y_test_proba_li, axis=1),
                                               'layer_{} - test.classifier_KAPPA'.format(layer_id)))
                train_aa_list.append(train_aa_acc)
                test_aa_list.append(test_aa_acc)
                train_acc_list.append(train_avg_acc)
                test_acc_list.append(test_avg_acc)

                opt_layer_id = get_opt_layer_id(test_acc_list if stop_by_test else train_acc_list)
                # set opt_datas
                if opt_layer_id == layer_id:
                    opt_datas = [y_train_proba_all[opt_layer_id], y_test_proba_all[opt_layer_id]]
                # early stop
                if self.early_stopping_rounds > 0 and layer_id - opt_layer_id >= self.early_stopping_rounds:
                    # log and save final result (opt layer)
                    LOGGER.info("[Result][Optimal Level Detected] opt_layer_id={}, accuracy_train={:.2f}%, accuracy_test={:.2f}%, AA accuracy_train={:.2f}%, AA_accuracy_test={:.2f}%".format(
                            opt_layer_id, train_acc_list[opt_layer_id], test_acc_list[opt_layer_id],
                            np.mean(train_aa_list[opt_layer_id]), np.mean(test_aa_list[opt_layer_id])))
                    LOGGER.info("[Result][train AA]:" + str(train_aa_list[opt_layer_id]))
                    LOGGER.info("[Result][test AA]:" + str(test_acc_list[opt_layer_id]))
                    LOGGER.info("[Result][test KAPPA]:" + str(kappa_list[opt_layer_id]))
                    if self.data_save_dir is not None:
                        self.save_data(opt_layer_id, *(opt_datas[1:]))
                    return opt_datas
                # save opt data if needed
                if self.data_save_rounds > 0 and (layer_id + 1) % self.data_save_rounds == 0:
                    self.save_data(layer_id, *(opt_datas[1:]))
                # inc layer_id
                layer_id += 1
            opt_datas = [y_train_proba_all[-1], y_test_proba_all[-1]]
            # log and save final result (last layer)
            LOGGER.info("[Result][Reach Max Layer] max_layer_num={}, accuracy_train={:.2f}%, accuracy_test={:.2f}%, AA accuracy_train={:.2f}%, AA_accuracy_test={:.2f}%".format(
                    self.max_layers, train_acc_list[-1], test_acc_list[-1], np.mean(train_aa_list[-1]),
                    np.mean(test_aa_list[-1])))
            LOGGER.info("[Result][train AA]:" + str(train_aa_list[-1]))
            LOGGER.info("[Result][test AA]:" + str(test_aa_list[-1]))
            LOGGER.info("[Result][test KAPPA]:" + str(kappa_list[-1]))
            if self.data_save_dir is not None:
                self.save_data(self.max_layers - 1, *(opt_datas[1:]))
            return opt_datas
        except KeyboardInterrupt:
            pass

    def save_data(self, layer_id, X_train, y_train, X_test, y_test, train_aa, test_aa, train_acc, test_acc, kappa, all_feature_importance):
        for pi, phase in enumerate(["train", "test"]):
            data_path = osp.join(self.data_save_dir, "layer_{}-{}.pkl".format(layer_id, phase))
            check_dir(data_path)
            data = {"X": X_train, "y": y_train, "train_AA": train_aa, "train_OA": train_acc, 'Feature_importance': all_feature_importance} if pi == 0 else {"X": X_test, "y": y_test, "test_AA":test_aa, "test_OA":test_acc, "KAPPA":kappa}
            LOGGER.info("Saving Data in {} ... X.shape={}, y.shape={}".format(data_path, data["X"].shape, data["y"].shape))
            with open(data_path, "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
