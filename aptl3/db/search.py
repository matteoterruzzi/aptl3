import collections
import json
import logging
import sqlite3
from time import perf_counter
from typing import Tuple, Optional

import numpy as np

from .procrustean import ProcrusteanDatabase


class SearchDatabase(ProcrusteanDatabase):
    """
    Cross-modal search and presentation of results.
    """

    def __init__(self, data_dir: Optional[str] = None):
        super().__init__(data_dir)
        self.__logger = logging.getLogger('search')

    def search(self, q: str, *, n: int = 50, search_k: int = -1) -> Tuple[int, int]:
        start_time = perf_counter()

        try:
            query_media_id: bytes = bytes.fromhex(q)
        except ValueError:
            _url = 'data:,' + q
            query_media_id: bytes = self.url_to_media_id(_url)

            def _v_gen():
                self.__logger.debug(f'Looking up by sentence embedding ...')
                x_embedding_id = self.get_embedding_id('sentence')
                x_embedding = self.get_embedding(x_embedding_id)
                x_path = [f'{x_embedding_id=:d}']
                x = x_embedding.transform(url=_url)
                self.__logger.debug(f'Computed embedding vector ({x_embedding_id=})')

                # yield x_embedding_id, x, x_path, 0  # NOTE: don't show similar sentences for this kind of search
                yield from _gpa_gen(x_embedding_id, x, x_path, 0)
        else:
            def _v_gen():
                self.__logger.debug(f'Looking up by media_id={q} ...')
                for x_embedding_id, x in self.get_media_vectors(query_media_id):
                    self.__logger.debug(f'Retrieved embedding vector ({x_embedding_id=})')
                    x = np.asarray(x)
                    x_path = [f'{x_embedding_id=:d}']
                    yield x_embedding_id, x, x_path, 0
                    yield from _gpa_gen(x_embedding_id, x, x_path, 0)

        def _gpa_gen(x_embedding_id, x, x_path, add_dis):
            self.__logger.debug(f'Looking up GPA models to expand search ...')
            _c = self.execute(
                'SELECT gpa_id FROM GeneralizedProcrustesAnalysis '
                'NATURAL JOIN OrthogonalProcrustesModel '
                'WHERE embedding_id = ? '
                'GROUP BY gpa_id', (x_embedding_id,))
            for gpa_id, in _c:
                gpa_path = x_path + [f'{gpa_id=:d}']
                try:
                    gpa = self.load_generalized_procrustes_analysis(gpa_id)
                    for y_embedding_id in gpa.orthogonal_models:
                        if y_embedding_id != x_embedding_id:
                            y_path = gpa_path + [f'{y_embedding_id=}']
                            self.__logger.debug(f'Expanding search via GPA #{gpa_id} {y_embedding_id=} ...')
                            try:
                                y = gpa.predict(x_embedding_id, y_embedding_id, x.reshape([1, -1])).reshape([-1])
                                yield y_embedding_id, y, y_path, add_dis + gpa.procrustes_distance
                            except Exception:
                                self.__logger.debug(f"Problems with {y_path=}", exc_info=True)
                except Exception:
                    self.__logger.debug(f"Problems with {gpa_path=}", exc_info=True)

        manifold_counter = collections.Counter()

        def _r_gen():
            relation_hits = []
            for embedding_id, v, path, add_dis in _v_gen():
                self.__logger.debug(f'-' * 80)
                self.__logger.debug(f'Searching ANN indexes of {embedding_id=} ({v.shape=} {path=} {add_dis=:.2f}) ...')
                for scan_count, (manifold_id, item_i, dis_i) in enumerate(self.find_vector_neighbours(
                        embedding_id, v, n=n, search_k=search_k)):
                    _c = self.execute(
                        'SELECT media_id, url FROM MediaLocations NATURAL JOIN ManifoldItems '
                        'WHERE manifold_id = ? AND item_i = ?', (manifold_id, item_i))
                    for media_id, url in _c:
                        rank = (dis_i + add_dis)  # * (len(path) + 1)  # very arbitrary
                        manifold_counter.update([manifold_id])
                        params = dict()
                        params['results_id'] = results_id
                        params['media_id'] = media_id
                        params['rank'] = rank
                        params['info'] = json.dumps({
                            'manifold_id': manifold_id,
                            'item_i': item_i,
                            'path': path,
                            'dis_i': dis_i,
                            'add_dis': add_dis,
                        })
                        self.__logger.debug(f'{rank=:.2f} {manifold_id=} {dis_i=:.2f} {media_id.hex()} {url}')

                        if media_id == query_media_id:
                            self.__logger.info(f'Query media found in ANN index search with distance {dis_i:.2f}')

                        _c2 = self.execute('SELECT relation_id, metadata '
                                           'FROM MediaRelations NATURAL JOIN Relations '
                                           'WHERE (media_id = :m AND other_media_id = :qm) '
                                           'OR (media_id = :qm AND other_media_id = :m) ',
                                           {'m': media_id, 'qm': query_media_id})
                        for relation_id, r_metadata in _c2:
                            self.__logger.debug(f'Relation found! {scan_count=} {relation_id=} {r_metadata}')
                            relation_hits.append(scan_count)

                        yield params
                        break
                    else:
                        self.__logger.warning(f'No location for {manifold_id=} {item_i=} {dis_i=:.2f} {len(path)=}')
            if relation_hits:
                self.__logger.debug(f'Related media found at {relation_hits}')

        with self._db:
            # self.__logger.debug(f'{self._db.in_transaction=}')
            self.begin_exclusive_transaction()
            # self.__logger.debug(f'{self._db.in_transaction=}')

            metadata = dict(
                q=q,
                n=n,
                search_k=search_k,
                query_media_id=query_media_id.hex(),
            )
            c = self._db.execute('INSERT INTO results.Results (metadata) VALUES (?);',
                                 (json.dumps(metadata),))
            results_id: int = c.lastrowid
            self.__logger.info(f'{results_id=} {metadata=}')

            try:
                self._db.executemany('INSERT INTO results.ResultsMedia (results_id, media_id, rank, info) '
                                     'VALUES (:results_id, :media_id, :rank, :info)', _r_gen())
            except sqlite3.OperationalError as ex:
                self.__logger.warning(f'{type(ex).__name__}: {str(ex)}')
                inserted = 0
            else:
                c = self.execute('SELECT COUNT(*) FROM results.ResultsMedia WHERE results_id = ?', (results_id,))
                inserted = c.fetchone()[0]

            # self.__logger.debug(f'{self._db.in_transaction=}')
            self._db.commit()
            # self.__logger.debug(f'{self._db.in_transaction=}')

        elapsed_ms = (perf_counter() - start_time) * 1000.
        self.__logger.info(f'{results_id=} {inserted=} manifold_id {manifold_counter} {elapsed_ms=:.1f}')
        return results_id, inserted
