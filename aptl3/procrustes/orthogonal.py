import warnings

import numpy as np
from scipy.linalg import orthogonal_procrustes


class OrthogonalProcrustesModel:
    # NOTE: A lot of redundant asserts and checks

    def pad(self, v):
        dim = v.shape[1]
        assert dim in [self.src_dim, self.dest_dim]
        assert dim <= self.work_dim
        if dim != self.work_dim:
            v = np.pad(v, ((0, 0), (0, self.work_dim - dim)), mode='wrap')
        v /= np.sum(v ** 2, axis=1, keepdims=True) ** .5
        return v

    def __init__(self, src_dim, dest_dim):
        self.src_dim = src_dim
        self.dest_dim = dest_dim
        self.work_dim = max(src_dim, dest_dim)
        if self.dest_dim < self.work_dim:
            warnings.warn('Destination vector space is smaller than work space.')  # this must be impossible in GPA.
        self.R, self.scale = None, None

    def fit(self, x, y):
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == self.src_dim
        assert y.shape[1] == self.dest_dim
        self.R, self.scale = orthogonal_procrustes(self.pad(x), self.pad(y))
        assert self.R.shape[0] == self.R.shape[1] == self.work_dim
        if self.dest_dim < self.work_dim:
            self.R = self.R[:, :self.dest_dim]
            # NOTE: there's a warning for the above dimension cut, which is not done in GPA.
        else:
            assert self.dest_dim == self.work_dim
        return self

    def predict(self, x):
        if self.R is None:
            raise RuntimeError("Did you fit the model first?")
        w = self.pad(x) @ self.R
        assert w.shape[1] == self.dest_dim
        w /= np.sum(w ** 2, axis=1, keepdims=True) ** .5
        return w

    transform = predict

    def inverse_transform(self, y):
        if self.R is None:
            raise RuntimeError("Did you fit the model first?")
        assert y.shape[1] == self.dest_dim
        assert self.R.shape[1] == self.dest_dim
        inv = self.R.T  # R is orthogonal ==> inverse == transpose
        inv = inv[:, :self.src_dim]  # cut extra dims that are not part of source embedding
        w = y @ inv
        assert w.shape[1] == self.src_dim
        w /= np.sum(w ** 2, axis=1, keepdims=True) ** .5
        return w
