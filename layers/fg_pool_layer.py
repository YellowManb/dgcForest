# -*- coding:utf-8 -*-
"""
"""
import numpy as np
from .base_layer import BaseLayer
from ..utils.log_utils import get_logger

LOGGER = get_logger('gcforest_HSI_rw.layers.fg_pool_layer')

class FGPoolLayer(BaseLayer):
    def __init__(self, layer_config, data_cache):
        """
        Pooling Layer (MaxPooling, AveragePooling)
        """
        super(FGPoolLayer, self).__init__(layer_config, data_cache)
        self.win_x = self.get_value("win_x", None, int, required=True)
        self.win_y = self.get_value("win_y", None, int, required=True)
        self.pool_method = self.get_value("pool_method", "avg", basestring)
    def fit_transform(self, train_config):
        LOGGER.info("[data][{}] bottoms={}, tops={}".format(self.name, self.bottom_names, self.top_names))
        self._transform(["train", "test"])

    def _transform(self, phases):
        for ti, top_name in enumerate(self.top_names):
            LOGGER.info("[progress][{}] ti={}/{}, top_name={}".format(ti, self.name, len(self.top_names), top_name))
            for phase in phases:
                # check top cache
                if self.check_top_cache([phase], ti)[0]:
                    continue
                X = self.data_cache.get(phase, self.bottom_names[ti])
                LOGGER.info('[data][{},{}] bottoms[{}].shape={}'.format(self.name, phase, ti, X.shape))
                n, c, h, w = X.shape
                win_x, win_y = self.win_x, self.win_y
                nh = (h - 1) / win_y + 1
                nw = (w - 1) / win_x + 1
                X_pool = np.empty((n, c, nh, nw), dtype=np.float32)
                if self.pool_method == 'max' or self.pool_method == 'avg':
                    for k in range(c):
                        for di in range(nh):
                            for dj in range(nw):
                                si = di * win_y
                                sj = dj * win_x
                                src = X[:, k, si:si + win_y, sj:sj + win_x]
                                src = src.reshape((X.shape[0], -1))
                                if self.pool_method == 'max':
                                    X_pool[:, k, di, dj] = np.max(src, axis=1)
                                elif self.pool_method == 'avg':
                                    X_pool[:, k, di, dj] = np.mean(src, axis=1)
                                else:
                                    raise ValueError('Unkown Pool Method, pool_method={}'.format(self.pool_method))
                                    # print ('\n')
                elif self.pool_method == 'l0' or self.pool_method == 'l2' or self.pool_method == 'l02':
                    for di in range(nh):
                        for dj in range(nw):
                            si = di * win_y
                            sj = dj * win_x
                            src = X[:, :, si:si + win_y, sj:sj + win_x]
                            src = src.reshape((X.shape[0], X.shape[1], -1))
                            src_size = src.shape[-1]
                            if self.pool_method == 'l0':
                                for i_samples in range(n):
                                    l0_number = np.zeros((src_size, 1))[:, 0]
                                    for i_src_size in range(src_size):
                                        temp_src = src[i_samples, :, i_src_size]
                                        l0_number[i_src_size] = np.linalg.norm(temp_src, ord=0)
                                    X_pool[i_samples, :, di, dj] = src[i_samples, :, np.argmin(l0_number)]
                            elif self.pool_method == 'l2':
                                for i_samples in range(n):
                                    l2_number = np.zeros((src_size, 1))[:, 0]
                                    for i_src_size in range(src_size):
                                        temp_src = src[i_samples, :, i_src_size]
                                        l2_number[i_src_size] = np.linalg.norm(temp_src, ord=2)
                                    X_pool[i_samples, :, di, dj] = src[i_samples, :, np.argmax(l2_number)]
                            elif self.pool_method == 'l02':
                                for i_samples in range(n):
                                    l0_number = np.zeros((src_size, 1), dtype=np.int8)[:, 0]
                                    for i_src_size in range(src_size):
                                        temp_src = src[i_samples, :, i_src_size]
                                        l0_number[i_src_size] = np.linalg.norm(temp_src, ord=0)
                                    maxl0 = np.min(l0_number)
                                    if maxl0*1.0 == 1.0:
                                        X_pool[i_samples, :, di, dj] = src[i_samples, :, np.argmin(l0_number)]
                                    elif sum(l0_number == maxl0) > 1:
                                        index = np.where(l0_number == maxl0)
                                        index = index[0]
                                        l0_max = src[i_samples, :, index]
                                        l0_max_l2 = np.zeros((len(l0_max), 1), dtype=np.float32)[:, 0]
                                        for j in range(len(l0_max)):
                                            l0_max_l2[j] = np.linalg.norm(l0_max[j], ord=2)
                                        l2_max_index = index[np.argmax(l0_max_l2)]
                                        X_pool[i_samples, :, di, dj] = src[i_samples, :, l2_max_index]
                                    else:
                                        X_pool[i_samples, :, di, dj] = src[i_samples, :, np.argmin(l0_number)]
                            else:
                                raise ValueError('Unkown Pool Method, pool_method={}'.format(self.pool_method))
                else:
                    raise ValueError('Unkown Pool Method, pool_method={}'.format(self.pool_method))
                LOGGER.info('[data][{},{}] tops[{}].shape={}'.format(self.name, phase, ti, X_pool.shape))
                self.data_cache.update(phase, top_name, X_pool)
