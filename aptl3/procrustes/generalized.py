from typing import Dict, Optional
import logging

import numpy as np

from .orthogonal import OrthogonalProcrustesModel


class GeneralizedProcrustesAnalysis:
    """https://en.wikipedia.org/wiki/Generalized_Procrustes_analysis"""

    def __init__(self, *, max_iter: int = 10):
        self.dim: Optional[int] = None
        self.procrustes_distance: Optional[float] = None
        self.n_samples: Optional[int] = None
        self.orthogonal_models: Dict[int, OrthogonalProcrustesModel] = dict()  # map embedding_id --> model
        self.max_iter: int = max_iter
        assert max_iter >= 2

    def fit(self, aligned_embedding_samples: Dict[int, np.ndarray]) -> None:
        assert self.max_iter >= 2  # otherwise one of the models will be left undefined
        assert len(set(samples.shape[0] for samples in aligned_embedding_samples.values())) == 1, "alignment error"
        dims: Dict[int, int] = {e: samples.shape[1] for e, samples in aligned_embedding_samples.items()}

        # take max dim manifold as reference
        reference_embedding_id: Optional[int] = max(dims.keys(), key=dims.get)  # this will become None after iter 0
        reference: np.ndarray = aligned_embedding_samples[reference_embedding_id]
        dim: int = reference.shape[1]

        # superimpose all other instances to current reference shape
        models: Dict[int, OrthogonalProcrustesModel] = dict()
        procrustes_distance: Optional[float] = None
        i: int = -1
        for i in range(self.max_iter):
            mean = np.zeros_like(reference)
            for embedding_id, src_dim in dims.items():
                x = aligned_embedding_samples[embedding_id]
                if embedding_id == reference_embedding_id:
                    logging.debug(f'Using embedding #{embedding_id} as reference with {dim=} ...')
                    # R would be the identity matrix
                    mean += x
                else:
                    logging.debug(f'Fitting orthogonal procrustes for embedding #{embedding_id} {src_dim=} ...')
                    model = OrthogonalProcrustesModel(src_dim, dim)
                    models[embedding_id] = model
                    model.fit(x, reference)
                    logging.debug(f'Fitted with {model.scale/reference.shape[0]=:.2%} ...')
                    y = model.predict(x)
                    mean += y

            # compute the mean shape of the current set of superimposed shapes
            mean /= len(aligned_embedding_samples)

            old_procrustes_distance = procrustes_distance
            procrustes_distance = np.linalg.norm(mean - reference) / np.sqrt(mean.shape[0])

            # take as new reference the average shape along the axis of the manifolds and repeat until convergence
            reference_embedding_id = None
            reference = mean
            logging.debug(f'Done GPA iteration #{i} ({procrustes_distance=:.2%}) ...')
            if old_procrustes_distance is not None and procrustes_distance / old_procrustes_distance >= .99:
                break
        assert procrustes_distance is not None
        logging.debug(f'GPA fitted ({procrustes_distance=:.2%} '
                      f'{"reached max_iter" if i >= self.max_iter-1 else "converged"}).')

        # store results only at end, to avoid inconsistent state
        self.dim = dim
        self.procrustes_distance = float(procrustes_distance)
        self.n_samples = reference.shape[0]
        self.orthogonal_models = models

    def predict(self, src_embedding_id: int, dest_embedding_id: int, x: np.ndarray):
        a = self.orthogonal_models[src_embedding_id]
        assert a.src_dim <= self.dim
        b = self.orthogonal_models[dest_embedding_id]
        assert b.src_dim <= self.dim
        assert a.dest_dim == b.dest_dim == self.dim
        assert x.shape[1] == a.src_dim
        y = a.transform(x)
        assert y.shape[0] == x.shape[0]
        assert y.shape[1] == a.dest_dim == b.dest_dim == self.dim
        z = b.inverse_transform(y)
        assert z.shape[0] == x.shape[0]
        assert z.shape[1] == b.src_dim
        return z
