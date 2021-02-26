import json
import logging
import re
import tempfile
from time import perf_counter
from typing import Tuple, Optional, Iterable, NamedTuple, Union, Any

import numpy as np

from .procrustean import ProcrusteanDatabase


class _EmbeddingNode(NamedTuple):
    embedding_id: int
    vector: Union[np.ndarray, Any]

    @property
    def rank(self):
        return 0

    def __repr__(self) -> str:
        r = self.__class__.__name__
        fmt = '(' + ', '.join(f'{name}=%r' for name in self._fields) + ')'
        r += fmt % self._replace(vector=self.vector.shape)
        return r


class _GPANode(NamedTuple):
    src: _EmbeddingNode
    gpa_id: int
    embedding_id: int
    vector: Union[np.ndarray, Any]
    dis: float

    @property
    def rank(self):
        return self.src.rank + self.dis

    def __repr__(self) -> str:
        r = self.__class__.__name__
        fmt = '(' + ', '.join(f'{name}=%r' for name in self._fields) + ')'
        r += fmt % self._replace(vector=self.vector.shape)
        return r


_SearchNode = Union[
    _EmbeddingNode,
    _GPANode,
    # given a node, we will finally retrieve a set of neighbour vectors
]


class SearchDatabase(ProcrusteanDatabase):
    """
    Cross-modal search and presentation of results.
    """

    def __init__(self, data_dir: Optional[str] = None):
        super().__init__(data_dir)
        self.__logger = logging.getLogger('search')
        self.__url_re = re.compile(  # I want it to be simple and not correct
            r'^(((http|ftp)s?|file)://?|data:[^ ],)[^ ]+$',
            re.IGNORECASE)
        self.__results_file = tempfile.NamedTemporaryFile(prefix=f'{__name__}.results.')
        self._audit_open_files_allowed.add(self.__results_file.fileno())
        self.__attach_results_db()

    def get_attach_results_db_query(self) -> str:
        # NOTE: I would like to use "file:results?mode=memory&cache=shared"
        #       but it seems buggy with QSqlite
        return f"ATTACH DATABASE 'file:{self.__results_file.name}' AS results;"

    @staticmethod
    def get_schema_results_db_script() -> str:
        return '''
            CREATE TABLE IF NOT EXISTS results.Results (
                results_id INTEGER PRIMARY KEY AUTOINCREMENT,
                metadata TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS results.ResultsMedia (
                results_id INTEGER NOT NULL REFERENCES Results ON DELETE CASCADE,
                media_id BLOB NOT NULL,
                info TEXT NULL,
                rank REAL NULL
            );
        '''

    def __attach_results_db(self):
        with self._db:
            self._db.executescript(self.get_attach_results_db_query())
            self._db.executescript(self.get_schema_results_db_script())
            c = self.execute("INSERT INTO results.Results (metadata) VALUES ('test')")
            self.__logger.debug(f'test results.Results id: {c.lastrowid!r}')

    def _gen_search_nodes(self, q: str) -> Iterable[_SearchNode]:
        q = q.strip()
        try:
            query_media_id: bytes = bytes.fromhex(q)
        except ValueError:
            if self.__url_re.match(q):
                self.__logger.debug('Looking up by url ...')
                for x_embedding_id, x in self.get_url_vectors(url=q):
                    self.__logger.debug(f'Computed embedding vector ({x_embedding_id=}) from url')
                    node = _EmbeddingNode(x_embedding_id, x)
                    yield node
                    yield from self._gen_gpa_expansions(node)
            else:
                _url = 'data:,' + q
                self.__logger.debug('Looking up by sentence embedding ...')
                x_embedding_id = self.get_embedding_id('sentence')
                x_embedding = self.get_embedding(x_embedding_id)
                x = x_embedding.transform(url=_url)
                self.__logger.debug(f'Computed embedding vector ({x_embedding_id=}) from sentence')
                node = _EmbeddingNode(x_embedding_id, x)
                # yield node  # NOTE: don't show similar sentences for this kind of search
                yield from self._gen_gpa_expansions(node)
        else:
            self.__logger.debug(f'Looking up by media_id={q} ...')
            for x_embedding_id, x in self.get_media_vectors(query_media_id):
                self.__logger.debug(f'Retrieved embedding vector ({x_embedding_id=}) from known media')
                x = np.asarray(x)
                node = _EmbeddingNode(x_embedding_id, x)
                yield node
                yield from self._gen_gpa_expansions(node)

    def _gen_gpa_expansions(self, x_node: _EmbeddingNode) -> Iterable[_GPANode]:
        self.__logger.debug('Looking up GPA models to expand search ...')
        for gpa_id in self.list_generalized_procrustes_analyzes(src_embedding_id=x_node.embedding_id):
            gpa = self.load_generalized_procrustes_analysis(gpa_id)
            for y_embedding_id in gpa.orthogonal_models:
                if y_embedding_id == x_node.embedding_id:
                    continue
                self.__logger.debug(f'Expanding search via GPA #{gpa_id} {y_embedding_id=} ...')
                y = gpa.predict(x_node.embedding_id, y_embedding_id,
                                x_node.vector.reshape([1, -1])).reshape([-1])
                yield _GPANode(x_node, gpa_id, y_embedding_id, y, gpa.procrustes_distance)

    def __gen_rows(self, *, results_id: int, nodes: Iterable[_SearchNode], n: int, search_k: int) -> Iterable[dict]:
        for node in nodes:
            self.__logger.debug(f'Searching ANN indexes of {repr(node)} ...')
            for manifold_id, item_i, dis_i in self.find_vector_neighbours(
                    node.embedding_id, node.vector, n=n, search_k=search_k):
                params = dict()
                params['results_id'] = results_id
                params['manifold_id'] = manifold_id
                params['item_i'] = item_i
                params['rank'] = node.rank + dis_i
                params['info'] = json.dumps({
                    'node': repr(node),
                    'manifold_id': manifold_id,
                    'item_i': item_i,
                    'dis_i': dis_i,
                })
                yield params

    def search(self, q: str, *,
               n: int = 50, search_k: int = -1,
               ) -> Tuple[int, int]:

        start_time = perf_counter()

        nodes: Tuple[_SearchNode, ...] = tuple(self._gen_search_nodes(q))

        nodes_time = perf_counter()
        nodes_elapsed_ms = (nodes_time - start_time) * 1000.
        self.__logger.info(f'Going to start from {len(nodes):d} search nodes. {nodes_elapsed_ms=:.1f}')

        with self._db:
            self.begin_exclusive_transaction()

            metadata = dict(
                q=q,
                n=n,
                search_k=search_k,
                nodes=repr(nodes),
            )
            c = self._db.execute('INSERT INTO results.Results (metadata) VALUES (?);',
                                 (json.dumps(metadata),))
            results_id: int = c.lastrowid
            self.__logger.info(f'{results_id=} {metadata=}')

            rows = self.__gen_rows(results_id=results_id, nodes=nodes, n=n, search_k=search_k)

            self._db.executemany(
                'INSERT INTO results.ResultsMedia (results_id, media_id, rank, info) '
                'VALUES ('
                '  :results_id, '
                '  (SELECT media_id FROM ManifoldItems '
                '   WHERE manifold_id = :manifold_id AND item_i = :item_i), '
                '  :rank, :info)', rows)

            c = self.execute('SELECT COUNT(*) FROM results.ResultsMedia WHERE results_id = ?', (results_id,))
            inserted = c.fetchone()[0]

            self._db.commit()

        ann_elapsed_ms = (perf_counter() - nodes_time) * 1000.
        elapsed_ms = (perf_counter() - start_time) * 1000.
        self.__logger.info(f'{results_id=} {inserted=} {nodes_elapsed_ms=:.1f} {ann_elapsed_ms=:.1f} {elapsed_ms=:.1f}')
        return results_id, inserted
