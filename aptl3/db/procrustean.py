import json
import logging
from typing import Tuple, List, Dict, Iterable, Optional

import numpy as np

from .relations import RelationsDatabase
from .manifolds import ManifoldsDatabase
from ..procrustes.orthogonal import OrthogonalProcrustesModel
from ..procrustes.generalized import GeneralizedProcrustesAnalysis


class ProcrusteanDatabase(ManifoldsDatabase, RelationsDatabase):

    def __init__(self, data_dir: Optional[str] = None):
        super().__init__(data_dir)
        self.__last_src_embedding_ids: Optional[Tuple[int, ...]] = None
        self.__last_alignments: Optional[List[Tuple[Tuple[int, int], ...]]] = None
        self.__last_aligned_embedding_samples: Optional[Dict[int, np.ndarray]] = None
        self.__last_gpa_id: Optional[int] = None
        self.__last_gpa: Optional[GeneralizedProcrustesAnalysis] = None

    def list_generalized_procrustes_analyzes(self, src_embedding_id: Optional[int] = None) -> List[int]:
        if src_embedding_id is None:
            c = self.execute('SELECT gpa_id FROM GeneralizedProcrustesAnalysis')
        else:
            c = self.execute(
                'SELECT gpa_id FROM GeneralizedProcrustesAnalysis '
                'NATURAL JOIN OrthogonalProcrustesModel '
                'WHERE embedding_id = ? '
                'GROUP BY gpa_id',
                (src_embedding_id,))
        return [row[0] for row in c]

    def load_orthogonal_procrustes_model(self, gpa_id: int, embedding_id: int) -> OrthogonalProcrustesModel:
        c = self._db.execute(
            'SELECT E.dim, GPA.dim, OPM.scale, OPM.orthogonal_matrix '
            'FROM OrthogonalProcrustesModel as OPM '
            'JOIN GeneralizedProcrustesAnalysis as GPA ON GPA.gpa_id = OPM.gpa_id '
            'JOIN Embeddings E ON E.embedding_id = OPM.embedding_id '
            'WHERE OPM.gpa_id = ? AND OPM.embedding_id = ?',
            (gpa_id, embedding_id))
        res = c.fetchone()
        if res is None:
            raise KeyError(f'No instance of OrthogonalProcrustesModel for {gpa_id=:d}, {embedding_id=:d}.')
        src_dim, gpa_dim, scale, matrix_data = res
        assert isinstance(src_dim, int)
        assert isinstance(scale, float)
        assert isinstance(matrix_data, bytes)
        model = OrthogonalProcrustesModel(src_dim, gpa_dim)
        matrix: np.ndarray = np.frombuffer(matrix_data, np.float64).reshape([model.work_dim, gpa_dim])
        model.R = matrix
        model.scale = scale
        return model

    def load_generalized_procrustes_analysis(self, gpa_id: int) -> GeneralizedProcrustesAnalysis:
        """ NOTE: n_samples and max_iter will not be fetched. """
        c = self._db.execute(
            'SELECT GPA.dim, GPA.procrustes_distance, OPM.embedding_id '
            'FROM OrthogonalProcrustesModel as OPM '
            'JOIN GeneralizedProcrustesAnalysis as GPA ON GPA.gpa_id = OPM.gpa_id '
            'WHERE OPM.gpa_id = ?', (gpa_id,))
        gpa = GeneralizedProcrustesAnalysis()
        for dim, procrustes_distance, embedding_id in c:
            gpa.dim = dim
            gpa.procrustes_distance = procrustes_distance
            gpa.orthogonal_models[embedding_id] = self.load_orthogonal_procrustes_model(
                gpa_id=gpa_id, embedding_id=embedding_id)
        return gpa

    def find_aligned_items(self, *,
                           src_embedding_ids: Tuple[int, ...],
                           src_relation_ids: Iterable[int],
                           max_samples: Optional[int] = None,
                           shuffle: bool = True,
                           ) -> Iterable[Tuple[Tuple[int, int], ...]]:
        """
        Search for manifold items for each manifold of the given tuple
        aligned according to the given equivalence relations (or all if not specified).
        Alignment with more than 2 manifolds is done according to the tolerance closure of the union of the relations.
        The alignments are tuples of the same length and order of src_manifold_ids, generated in random order.
        Each element of the alignment tuple is itself a tuple of <manifold_id, item_i>.
        Any given item (or media) may be part of many different alignments.
        Reference to the media_id associated with an item is not useful and will NOT be fetched.
        :param src_embedding_ids: tuple of unique embedding ids
        :param src_relation_ids:
        :param max_samples: limit generated number of alignments
        :param shuffle: set to False if you prefer the samples ordered by the database
        :return: generator of alignments
        """
        assert len(src_embedding_ids) == len(set(src_embedding_ids)), src_embedding_ids

        selects = list()  # SELECT ...
        selects += [f"MI{m:d}.manifold_id" for m in src_embedding_ids]
        selects += [f"MI{m:d}.item_i" for m in src_embedding_ids]
        group_by = selects.copy()
        if shuffle:
            selects += ["random() as rand_"]

        tables = list()  # FROM ...
        tables += [f"ManifoldItems as MI{i:d}" for i in src_embedding_ids]
        tables += [f"Manifolds as M{i:d}" for i in src_embedding_ids]
        tables += [f"MediaRelations as MR{i:d}" for i in src_embedding_ids[:-1]]  # see how symmetric closure is done

        clauses = list()  # WHERE ...
        clauses += [f"M{i:d}.ready" for i in src_embedding_ids]
        clauses += [f"M{i:d}.embedding_id = {i:d}" for i in src_embedding_ids]
        clauses += [f"MI{i:d}.manifold_id = M{i:d}.manifold_id" for i in src_embedding_ids]
        clauses += ["(" + " OR ".join(f"MR{m:d}.relation_id = {e:d}" for e in src_relation_ids) + ")"
                    for m in src_embedding_ids[:-1]]

        for i, ml in enumerate(src_embedding_ids[:-1]):
            tol = list()  # tolerance relation with any of the successors
            # tol += [f"(MI{ml:d}.media_id = MI{mr:d}.media_id"
            #         f" AND MR{ml:d}.media_id = (SELECT media_id FROM MediaRelations"
            #         f"      WHERE relation_id = MR{ml:d}.relation_id LIMIT 1) "
            #         f" AND MR{ml:d}.other_media_id = (SELECT other_media_id FROM MediaRelations"
            #         f"      WHERE relation_id = MR{ml:d}.relation_id LIMIT 1))"
            #         for mr in src_embedding_ids[i + 1:]]  # reflexive (NOTE: must fix an arbitrary tuple of MR{ml}!)
            # FIXME: the reflexive closure (also without use of the relation) was killing the performance! => disabled
            tol += [f"(MI{ml:d}.media_id = MR{ml:d}.media_id AND MI{mr:d}.media_id = MR{ml:d}.other_media_id)"
                    for mr in src_embedding_ids[i + 1:]]  # from left to right
            tol += [f"(MI{ml:d}.media_id = MR{ml:d}.other_media_id AND MI{mr:d}.media_id = MR{ml:d}.media_id)"
                    for mr in src_embedding_ids[i + 1:]]  # from right to left
            assert len(tol) > 0
            clauses += ['(' + ' OR '.join(tol) + ')']

        query = "SELECT " + ", ".join(selects)
        query += " FROM " + " JOIN ".join(tables)
        query += " WHERE " + " AND ".join(clauses)
        query += " GROUP BY " + ", ".join(group_by)
        if shuffle:
            query += " ORDER BY rand_"
        if max_samples is not None:
            query += f" LIMIT {max_samples:d}"

        logging.debug(query)

        c = self._db.execute(query)
        for row_i, row in enumerate(c):
            assert len(row) == len(src_embedding_ids)*2 + (1 if shuffle else 0)
            assert all(isinstance(x, int) for x in row[:-1])
            alignment = []
            for i in range(len(src_embedding_ids)):
                alignment.append((row[i], row[len(src_embedding_ids) + i]))
            if row_i % 1000 == 0:
                logging.debug(f'{row_i=:d} {alignment=}')
            yield alignment

    def build_generalized_procrustes(self, *,
                                     src_embedding_ids: Iterable[int],
                                     src_relation_ids: Iterable[int],
                                     min_samples: int = 100,
                                     max_samples: int = 10000,
                                     keep_training_data: bool = False,
                                     ) -> Tuple[int, GeneralizedProcrustesAnalysis]:
        """
        Specify keep_training_data=True if you later need to evaluate on the same training set.

        # Procedure:
        # 1. collect sample of aligned manifold items of specified embeddings from db
        # 2. collect vector data from manifold indexes
        # 3. run Generalized Procrustes analysis
        # 4. insert gpa model into db
        """

        # 1. collect sample of aligned manifold items of specified embeddings from db
        src_embedding_ids = tuple(src_embedding_ids)
        assert len(src_embedding_ids) >= 2
        logging.info(f'Querying the database to align {len(src_embedding_ids):d} ...')
        alignments: List[Tuple[Tuple[int, int], ...]] = list(self.find_aligned_items(
            src_embedding_ids=src_embedding_ids, src_relation_ids=src_relation_ids, max_samples=max_samples))
        if len(alignments) < min_samples:
            raise RuntimeError(f'Not enough aligned samples, got {len(alignments)}')
        logging.info(f'Found {len(alignments)} aligned items ...')

        # 2. collect vector data in item order (to decrease cache invalidation) and then populate aligned 2D arrays
        aligned_embedding_samples: Dict[int, np.ndarray] = self.__collect_aligned_samples(
            src_embedding_ids=src_embedding_ids, alignments=alignments)

        # 3. GPA
        logging.info(f'Running Generalized Procrustes Analysis with {len(aligned_embedding_samples):d} embeddings ...')
        gpa = GeneralizedProcrustesAnalysis()
        gpa.fit(aligned_embedding_samples)

        # 4. store gpa model into db
        gpa_id, gpa = self.__insert_gpa(gpa)
        logging.info(f'Stored new Generalized Procrustes Model #{gpa_id}.')

        if keep_training_data:
            self.__last_src_embedding_ids = src_embedding_ids
            self.__last_alignments = alignments
            self.__last_aligned_embedding_samples = aligned_embedding_samples
            self.__last_gpa_id = gpa_id
            self.__last_gpa = gpa

        return gpa_id, gpa

    def __collect_aligned_samples(self, *,
                                  src_embedding_ids: Tuple[int, ...],
                                  alignments: List[Tuple[Tuple[int, int], ...]],
                                  ) -> Dict[int, np.ndarray]:
        """
        Collect vector data in item order (to decrease cache invalidation) and then populate aligned 2D arrays.
        The samples in the returned dictionary (keyed by embedding_id) is in the same order of the given alignments.
        """
        aligned_embedding_samples: Dict[int, np.ndarray] = dict()
        for ei, embedding_id in enumerate(src_embedding_ids):
            logging.info(f'Collecting vector data for {embedding_id=} ...')
            column_keys: List[Tuple[int, int]] = [al[ei] for al in alignments]
            values: Dict[Tuple[int, int], np.ndarray] = dict()
            for manifold_id, item_i in sorted(set(column_keys)):
                values[manifold_id, item_i] = np.asarray(self.get_item_vector(manifold_id, item_i), dtype=np.float32)
            aligned_embedding_samples[embedding_id] = np.vstack([values[k] for k in column_keys])
        return aligned_embedding_samples

    def __insert_gpa(self, gpa: GeneralizedProcrustesAnalysis) -> Tuple[int, GeneralizedProcrustesAnalysis]:
        metadata = dict(
            n_samples=gpa.n_samples,
            max_iter=gpa.max_iter,
        )
        with self._db:
            self.begin_exclusive_transaction()
            c = self._db.execute(
                'INSERT INTO GeneralizedProcrustesAnalysis (dim, procrustes_distance, metadata) VALUES (?, ?, ?);',
                (gpa.dim, float(gpa.procrustes_distance), json.dumps(metadata)))
            gpa_id: int = c.lastrowid
            logging.debug(f'Storing new Generalized Procrustes Model #{gpa_id}: '
                          f'{gpa.dim=}, {gpa.procrustes_distance=}, {metadata=}')

            for embedding_id, model in gpa.orthogonal_models.items():
                matrix: np.ndarray = model.R
                matrix_data = matrix.astype(np.float64).tobytes('C')
                assert isinstance(matrix_data, bytes)
                self._db.execute(
                    'INSERT INTO OrthogonalProcrustesModel (gpa_id, embedding_id, scale, orthogonal_matrix) '
                    'VALUES (?, ?, ?, ?)', (gpa_id, embedding_id, float(model.scale), matrix_data))
            self._db.commit()
        return gpa_id, gpa

    def evaluate_generalized_procrustes(
            self, *,
            gpa: Optional[GeneralizedProcrustesAnalysis] = None,
            src_relation_ids: Optional[Iterable[int]] = None,
            min_samples: int = 100,
            max_samples: int = 10000,
            max_query: int = 5000,
            search_n: int = 1000,
            search_k: int = -1,
            embedding_id_pairs: Optional[List[Tuple[int, int]]] = None,
            find_neighbours: bool = False,
            ) -> Iterable[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Evaluate a Generalized Procrustes Analysis model on the last used training set or on a provided validation set.
        You may have previously called build_generalized_procrustes with keep_training_data=True.

        :param gpa: the model to be evaluated. If not provided, then use the last one that was previously built.
        :param src_relation_ids: relations to be used to find alignments. If None, then use the last used to build.
        :param min_samples: minimum number of alignments
        :param max_samples: maximum number of alignments
        :param max_query: maximum number of alignments actually used
        :param search_n: number of items to be retrieved by the the ann engine
        :param search_k: passed to the ann engine
        :param embedding_id_pairs: (src, dest) pairs to evaluate the model on. Consider all N*(N-1) if not specified.
        :param find_neighbours: if False, then the yielded top_dis and ranks arrays are empty; Takes longer if True.
        :return: iterable of at most N*(N-1) tuples of (src_embedding_id, dest_embedding_id, ranks, top_dis, true_dis)
        """
        if gpa is None:
            if self.__last_gpa is None:
                raise RuntimeError('Last built GPA model is not available.')
            gpa = self.__last_gpa

        if src_relation_ids is None:
            if (self.__last_src_embedding_ids is None
                    or self.__last_alignments is None
                    or self.__last_aligned_embedding_samples is None):
                raise RuntimeError('Last used GPA training data is not available.')
            embedding_ids: Tuple[int, ...] = self.__last_src_embedding_ids
            alignments: List[Tuple[Tuple[int, int], ...]] = self.__last_alignments
            if len(alignments) > max_samples:
                raise RuntimeError('Specified max_samples is violated while using last used GPA training set.')
            aligned_embedding_samples: Dict[int, np.ndarray] = self.__last_aligned_embedding_samples
        else:
            embedding_ids: Tuple[int, ...] = tuple(gpa.orthogonal_models.keys())
            alignments: List[Tuple[Tuple[int, int], ...]] = list(self.find_aligned_items(
                src_embedding_ids=embedding_ids, src_relation_ids=src_relation_ids, max_samples=max_samples))
            aligned_embedding_samples: Dict[int, np.ndarray] = self.__collect_aligned_samples(
                src_embedding_ids=embedding_ids, alignments=alignments)

        if len(alignments) < min_samples:
            raise RuntimeError(f'Not enough aligned samples, got {len(alignments)}')

        alignments = alignments[:max_query]

        assert len(embedding_ids) == len(alignments[0]), (
            f"{embedding_ids=} {len(alignments[0])=}")
        assert all(eid in gpa.orthogonal_models for eid in embedding_ids), (
            f"{embedding_ids=} {gpa.orthogonal_models.keys()=}")
        assert all(eid in aligned_embedding_samples for eid in embedding_ids), (
            f"{embedding_ids=} {aligned_embedding_samples.keys()=}")

        for ei, src_embedding_id in enumerate(embedding_ids):
            x = aligned_embedding_samples[src_embedding_id][:max_query]
            for ej, dest_embedding_id in enumerate(embedding_ids):
                if ei == ej:
                    continue
                if embedding_id_pairs is not None and (src_embedding_id, dest_embedding_id) not in embedding_id_pairs:
                    continue

                y = aligned_embedding_samples[dest_embedding_id][:max_query]
                y_pred = gpa.predict(
                    src_embedding_id=src_embedding_id,
                    dest_embedding_id=dest_embedding_id,
                    x=x)

                assert x.shape[0] == y.shape[0] == y_pred.shape[0] == len(alignments), (
                    f"{x.shape=} {y.shape=} {y_pred.shape=} {len(alignments)=}")

                ranks: np.ndarray = np.empty(shape=x.shape[:1], dtype=np.int32)
                top_dis: np.ndarray = np.empty(shape=x.shape[:1], dtype=np.float32)

                if find_neighbours:
                    for i, v in enumerate(y_pred):
                        results: List[Tuple[int, int, float]] = list(self.find_vector_neighbours(
                            dest_embedding_id, v, n=search_n, search_k=search_k))
                        results.sort(key=lambda t: t[2])
                        if len(results) <= 0:
                            raise RuntimeError(f"No results for find_vector_neighbours with {dest_embedding_id=}")

                        top = np.asarray(self.get_item_vector(*results[0][:2]))
                        top_dis[i] = np.linalg.norm(top - y[i], ord=2)

                        needle: Tuple[int, int] = alignments[i][ej]

                        for rank, (manifold_id, item_i, dis) in enumerate(results):
                            if (manifold_id, item_i) == needle:
                                ranks[i] = rank
                                break
                        else:
                            ranks[i] = 1 << 30

                true_dis = np.linalg.norm(y - y_pred, ord=2, axis=1)
                assert tuple(true_dis.shape) == tuple(top_dis.shape)

                yield src_embedding_id, dest_embedding_id, ranks, top_dis, true_dis

    def forget_generalized_procrustes_training_data(self):
        self.__last_src_embedding_ids = None
        self.__last_alignments = None
        self.__last_aligned_embedding_samples = None
        self.__last_gpa_id = None
        self.__last_gpa = None
