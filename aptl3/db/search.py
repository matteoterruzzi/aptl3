import json
import logging
import re
import tempfile
from time import perf_counter
from typing import Tuple, Optional, Iterable, NamedTuple, Union, Any, List

import numpy as np

from .procrustean import ProcrusteanDatabase


def _search_node_repr(self: '_SearchNode') -> str:
    r = self.__class__.__name__
    fmt = '(' + ', '.join(f'{name}=%r' for name in self._fields) + ')'
    r += fmt % (self._replace(vector=self.vector.shape) if 'vector' in self._fields else self)
    return r


class _EmbeddingNode(NamedTuple):
    embedding_id: int
    vector: Union[np.ndarray, Any]

    @property
    def rank(self):
        return 0

    __repr__ = _search_node_repr


class _SameSpaceNode(NamedTuple):
    src: _EmbeddingNode
    embedding_id: int

    @property
    def vector(self) -> Union[np.ndarray, Any]:
        return self.src.vector

    @property
    def rank(self):
        return self.src.rank

    __repr__ = _search_node_repr


class _GPANode(NamedTuple):
    src: _EmbeddingNode
    gpa_id: int
    embedding_id: int
    vector: Union[np.ndarray, Any]
    dis: float

    @property
    def rank(self):
        return self.src.rank + self.dis

    __repr__ = _search_node_repr


_SearchNode = Union[
    _EmbeddingNode,
    _SameSpaceNode,
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

            CREATE TABLE IF NOT EXISTS results.ResultsItems (
                results_id INTEGER NOT NULL REFERENCES Results ON DELETE CASCADE,
                embedding_id INTEGER NOT NULL,
                manifold_id INTEGER NOT NULL,
                item_i INTEGER NOT NULL,
                node TEXT NOT NULL,
                dis_i REAL NOT NULL,
                rank REAL NOT NULL CHECK ( rank >= 0 )
            );
            
            CREATE TABLE IF NOT EXISTS results.ResultsMedia (
                results_id INTEGER NOT NULL REFERENCES Results ON DELETE CASCADE,
                media_id INTEGER NOT NULL,
                rank REAL NOT NULL CHECK ( rank >= 0 )
            );
            
            CREATE VIEW IF NOT EXISTS results.ResultsMediaFiltered (results_id, media_id, rank) AS
                WITH R_stats(results_id, min_rank, max_rank, rank_range) AS (
                    SELECT results_id, MIN(rank), MAX(rank), MAX(rank)-MIN(rank) 
                    FROM results.ResultsMedia GROUP BY results_id)
                SELECT RM.results_id, media_id, MIN(rank)
                FROM results.ResultsMedia AS RM 
                JOIN R_stats ON RM.results_id = R_stats.results_id
                WHERE rank - (R_stats.min_rank + 0.55) <= R_stats.rank_range / 2
                GROUP BY RM.results_id, media_id;
        '''

    def __attach_results_db(self):
        with self._db:
            self._db.executescript(self.get_attach_results_db_query())
            self._db.executescript(self.get_schema_results_db_script())
            c = self.execute("INSERT INTO results.Results (metadata) VALUES ('test')")
            self.__logger.debug(f'test results.Results id: {c.lastrowid!r}')

    def _gen_search_nodes(self, q: str) -> Iterable[_EmbeddingNode]:
        q = q.strip()
        try:
            query_media_id: bytes = bytes.fromhex(q)
        except ValueError:
            if self.__url_re.match(q):
                self.__logger.debug('Looking up by url ...')
                for x_embedding_id, x in self.get_url_vectors(url=q):
                    self.__logger.debug(f'Computed embedding vector ({x_embedding_id=}) from url')
                    yield _EmbeddingNode(x_embedding_id, x)
            else:
                _url = 'data:,' + q
                self.__logger.debug('Looking up by text ...')
                for x_embedding_id, x in self.get_url_vectors(url=_url):
                    self.__logger.debug(f'Computed embedding vector ({x_embedding_id=}) from text')
                    yield _EmbeddingNode(x_embedding_id, x)
        else:
            self.__logger.debug(f'Looking up by media_id={q} ...')
            for x_embedding_id, x in self.get_media_vectors(query_media_id):
                self.__logger.debug(f'Retrieved embedding vector ({x_embedding_id=}) from known media')
                x = np.asarray(x)
                yield _EmbeddingNode(x_embedding_id, x)

    def _gen_same_space_expansions(self, x_node: _EmbeddingNode) -> Iterable[_SameSpaceNode]:
        c = self._db.execute(
            'SELECT embedding_id FROM Embeddings WHERE space IS NOT NULL AND LENGTH(space) > 0 AND space = ('
            '   SELECT space FROM Embeddings WHERE embedding_id = ?)',
            (x_node.embedding_id,))
        for y_embedding_id, in c:
            if y_embedding_id == x_node.embedding_id:
                continue
            self.__logger.debug(f'Expanding search via same space {y_embedding_id=} ...')
            yield _SameSpaceNode(x_node, y_embedding_id)

    def _gen_gpa_expansions(self, x_node: _EmbeddingNode) -> Iterable[_GPANode]:
        for gpa_id in self.list_generalized_procrustes_analyzes(src_embedding_id=x_node.embedding_id):
            gpa = self.load_generalized_procrustes_analysis(gpa_id)
            for y_embedding_id in gpa.orthogonal_models:
                if y_embedding_id == x_node.embedding_id:
                    continue
                self.__logger.debug(f'Expanding search via GPA #{gpa_id} {y_embedding_id=} ...')
                y = gpa.predict(x_node.embedding_id, y_embedding_id,
                                x_node.vector.reshape([1, -1])).reshape([-1])
                yield _GPANode(x_node, gpa_id, y_embedding_id, y, gpa.procrustes_distance)

    def _filter_search_node(self, node: _SearchNode) -> bool:
        keep: bool = True  # self.get_embedding_tuple(node.embedding_id).modality != 'text'
        if not keep:
            self.__logger.debug(f'Leaving out {node!r} ...')
        return keep

    def __gen_rows(self, *, results_id: int, nodes: Iterable[_SearchNode], n: int, search_k: int) -> Iterable[dict]:
        for node in nodes:
            self.__logger.debug(f'Searching ANN indexes of {repr(node)} ...')
            for manifold_id, item_i, dis_i in self.find_vector_neighbours(
                    node.embedding_id, node.vector, n=n, search_k=search_k):
                params = dict()
                params['results_id'] = results_id
                params['embedding_id'] = node.embedding_id
                params['manifold_id'] = manifold_id
                params['item_i'] = item_i
                params['node'] = repr(node)
                params['dis_i'] = dis_i
                params['rank'] = node.rank + dis_i
                yield params

    def search(self, q: str, *,
               n: int = 50, search_k: int = -1,
               ) -> Tuple[int, int]:

        start_time = perf_counter()

        nodes: Tuple[_SearchNode, ...] = tuple(self._gen_search_nodes(q))

        expansions: List[_SearchNode] = []
        self.__logger.debug('Looking up same-space embeddings to expand search ...')
        for node in [*nodes, *expansions]:
            expansions.extend(self._gen_same_space_expansions(node))
        self.__logger.debug('Looking up GPA models to expand search ...')
        for node in [*nodes, *expansions]:
            expansions.extend(self._gen_gpa_expansions(node))

        nodes = tuple(filter(self._filter_search_node, [*nodes, *expansions]))

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
                'INSERT INTO results.ResultsItems (results_id, embedding_id, manifold_id, item_i, node, dis_i, rank) '
                'VALUES (:results_id, :embedding_id, :manifold_id, :item_i, :node, :dis_i, :rank)', rows)

            self._db.execute(
                'INSERT INTO results.ResultsMedia (results_id, media_id, rank) '
                'WITH RI_G_pre(embedding_id, rank_avg) AS ( '
                '   SELECT embedding_id, AVG(rank) '
                '   FROM results.ResultsItems '
                '   WHERE results_id = :results_id '
                '   GROUP BY embedding_id), '
                'RI_G(embedding_id, rank_low_avg) AS ( '
                '   SELECT RI_G_pre.embedding_id, AVG(rank) + 0.0001 '
                '   FROM results.ResultsItems AS RI '
                '   JOIN RI_G_pre ON RI_G_pre.embedding_id = RI.embedding_id '
                '   WHERE results_id = :results_id AND rank <= RI_G_pre.rank_avg '
                '   GROUP BY RI_G_pre.embedding_id) '
                'SELECT RI.results_id, MI.media_id, RI.rank / RI_G.rank_low_avg '
                'FROM results.ResultsItems AS RI '
                'JOIN ManifoldItems MI ON RI.manifold_id = MI.manifold_id AND RI.item_i = MI.item_i '
                'JOIN RI_G ON RI_G.embedding_id = RI.embedding_id '
                'WHERE RI.results_id = :results_id', dict(results_id=results_id))

            c = self.execute('SELECT COUNT(*) FROM results.ResultsMedia WHERE results_id = ?', (results_id,))
            inserted = c.fetchone()[0]

            for row in self._tabulate_search_results_stats(results_id):
                self.__logger.debug(' '.join(row))

            self._db.commit()

        ann_elapsed_ms = (perf_counter() - nodes_time) * 1000.
        elapsed_ms = (perf_counter() - start_time) * 1000.
        self.__logger.info(f'{results_id=} {inserted=} {nodes_elapsed_ms=:.1f} {ann_elapsed_ms=:.1f} {elapsed_ms=:.1f}')
        return results_id, inserted

    def clear_search_results(self, results_id: int):
        self._db.execute('DELETE FROM results.ResultsItems WHERE results_id = ?', (results_id,))
        self._db.execute('DELETE FROM results.ResultsMedia WHERE results_id = ?', (results_id,))

    def _tabulate_search_results_stats(self, results_id: int) -> Iterable[Iterable[str]]:
        c = self.execute(
            'WITH RI_G_pre(embedding_id, rank_avg) AS ( '
            '   SELECT embedding_id, AVG(rank) '
            '   FROM results.ResultsItems '
            '   WHERE results_id = :results_id '
            '   GROUP BY embedding_id), '
            'RI_G(embedding_id, rank_low_avg) AS ( '
            '   SELECT RI_G_pre.embedding_id, AVG(rank) + 0.0001 '
            '   FROM results.ResultsItems AS RI '
            '   JOIN RI_G_pre ON RI_G_pre.embedding_id = RI.embedding_id '
            '   WHERE results_id = :results_id AND rank <= RI_G_pre.rank_avg '
            '   GROUP BY RI_G_pre.embedding_id) '
            'SELECT E.modality, RI.embedding_id, manifold_id, COUNT(*), '
            '   MIN(rank / RI_G.rank_low_avg), '
            '   AVG(rank / RI_G.rank_low_avg), '
            '   MAX(rank / RI_G.rank_low_avg), '
            '   MIN(dis_i), AVG(dis_i), MAX(dis_i) '
            'FROM results.ResultsItems AS RI '
            'JOIN RI_G ON RI_G.embedding_id = RI.embedding_id '
            'NATURAL JOIN Embeddings AS E '
            'WHERE results_id = :results_id '
            'GROUP BY manifold_id '
            'ORDER BY E.modality, RI.embedding_id, manifold_id', dict(results_id=results_id))
        yield [
            'embedding_id', 'manifold_id', 'COUNT(*)',
            'MIN(rank)', 'AVG(rank)', 'MAX(rank)',
            'MIN(dis_i)', 'AVG(dis_i)', 'MAX(dis_i)',
            'modality'
        ]
        for _mod, _e_id, _m_id, count, min_rank, avg_rank, max_rank, min_dis_i, avg_dis_i, max_dis_i in c:
            yield [
                f'{_e_id:12d}', f'{_m_id:11d}', f'{count:8d}',
                f'{min_rank:9.4f}', f'{avg_rank:9.4f}', f'{max_rank:9.4f}',
                f'{min_dis_i:10.4f}', f'{avg_dis_i:10.4f}', f'{max_dis_i:10.4f}',
                f'{_mod}'
            ]
