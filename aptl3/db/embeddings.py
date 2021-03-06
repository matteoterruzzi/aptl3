import logging
from contextlib import nullcontext
from functools import lru_cache
from typing import Optional, Dict, Tuple, Any, Iterable, NamedTuple

from .schema import SchemaDatabase
from ..embedding.abc import Embedding, RequestIgnored


# TODO: add tools to compute embeddings in a separate process, exploiting their static nature, or passing the data dir.
#       Maybe bring here a generalized version of what is defined in manifolds.py


class DeprecatedEmbeddingVersionException(Exception):
    """This exception is managed by EmbeddingsDatabase.get_embedding_id(name)"""
    pass


class EmbeddingTuple(NamedTuple):
    embedding_id: int
    dim: int
    name: str
    version: str
    space: str
    modality: str
    ready: bool
    embedding: Optional[Embedding]


class EmbeddingsDatabase(SchemaDatabase):
    """
    Static naming of embedding models and versioning (or future support for complex configurations)
    """

    def __init__(self, data_dir: Optional[str] = None):
        super().__init__(data_dir)
        self.__logger = logging.getLogger('embeddings')
        # will map embedding id to name and instance
        self.__embeddings: Dict[int, EmbeddingTuple] = dict()

    @classmethod
    def initialize_static_embeddings(cls):
        for name in [
            'image',
            'sentence',
        ]:
            cls.make_static_embedding_by_name(name)

    @classmethod
    @lru_cache(maxsize=None)
    def make_static_embedding_by_name(cls, name: str, data_dir: Optional[str] = None) -> Embedding:
        if name.startswith('r-'):
            from ..embedding.random import NonLSHEmbedding
            dim = int(name[2:])
            return NonLSHEmbedding(dim)
        elif name == 'sentence':
            from ..embedding.sentence import SentenceEmbedding
            return SentenceEmbedding()
        elif name == 'image':
            from ..embedding.image import ImageEmbedding
            return ImageEmbedding(data_dir=data_dir)
        elif name == 'clip-text':
            from ..embedding.openai_clip import ClipTextEmbedding
            return ClipTextEmbedding()
        elif name == 'clip-image':
            from ..embedding.openai_clip import ClipImageEmbedding
            return ClipImageEmbedding()
        else:
            raise KeyError(f'No embedding runtime available for {name}.')

    def get_embedding_tuple(self, embedding_id: int) -> EmbeddingTuple:
        try:
            return self.__embeddings[embedding_id]
        except KeyError:
            c = self.execute(
                'SELECT dim, name, version, space, modality, ready '
                'FROM Embeddings WHERE embedding_id = ?', (embedding_id,))
            row = c.fetchone()
            if row is None:
                raise ValueError(f'Embedding #{embedding_id} not found in db.')
            dim, name, version, space, modality, ready = row
            _tuple = EmbeddingTuple(
                embedding_id,
                int(dim), name, version, space, modality, bool(ready),
                None)
            self.__embeddings[embedding_id] = _tuple
            return _tuple

    def get_embedding(self, embedding_id: int) -> Embedding:
        _tuple = self.get_embedding_tuple(embedding_id)
        if _tuple.embedding is not None:
            return _tuple.embedding
        embedding = self.make_static_embedding_by_name(_tuple.name, data_dir=self.get_data_dir())
        self.__embeddings[embedding_id] = _tuple._replace(embedding=embedding)
        if embedding.get_version() != _tuple.version:
            raise DeprecatedEmbeddingVersionException(f"The model {_tuple.version:s} has been deprecated.")
        return embedding

    def get_embedding_id(self, name: str, *, commit: bool = True) -> int:
        for _tuple in self.__embeddings.values():
            if _tuple.name == name:
                return _tuple.embedding_id

        c = self.execute('SELECT embedding_id FROM Embeddings WHERE name = ?', (name,))
        row = c.fetchone()
        if row is None:
            return self.add_embedding(name=name, commit=commit)
        else:
            _embedding_id: int = row[0]
            try:
                self.get_embedding(_embedding_id)
            except DeprecatedEmbeddingVersionException:
                return self.push_new_embedding_version(name=name, commit=commit)
            return _embedding_id

    def push_new_embedding_version(self, name: str, *, commit: bool = True) -> int:
        with (self._db if commit else nullcontext()):
            c = self.execute('SELECT embedding_id FROM Embeddings WHERE name = ?', (name,))
            row = c.fetchone()
            if row is None:
                raise ValueError(f'No embedding for {name}')
            old_embedding_id: int = row[0]
            deprecation_name: str = f'DEPRECATED-{old_embedding_id:d}-{name}'
            self.execute('UPDATE Embeddings SET name = :name, ready = 0 '
                         'WHERE embedding_id = :embedding_id',
                         {'embedding_id': old_embedding_id,
                          'name': deprecation_name})
            self.__embeddings[old_embedding_id] = self.__embeddings[old_embedding_id]._replace(
                name=deprecation_name)

            return self.add_embedding(name, commit=False)

    def add_embedding(self, name: str, *, commit: bool = True) -> int:
        new_embedding: Embedding = self.make_static_embedding_by_name(name, data_dir=self.get_data_dir())
        dim = new_embedding.get_dim()
        version = new_embedding.get_version()
        space = new_embedding.get_space()
        modality = new_embedding.get_modality()
        with (self._db if commit else nullcontext()):
            c = self.execute('INSERT INTO Embeddings (dim, name, version, space, modality, ready) '
                             'VALUES (?, ?, ?, ?, ?, 1)', (dim, name, version, space, modality))
            new_embedding_id: int = c.lastrowid
            self.__embeddings[new_embedding_id] = EmbeddingTuple(
                embedding_id=new_embedding_id,
                dim=dim,
                name=name,
                version=version,
                space=space,
                modality=modality,
                ready=True,
                embedding=new_embedding,
            )
            return new_embedding_id

    def list_ready_embedding_ids(self) -> Iterable[int]:
        c = self.execute('SELECT embedding_id FROM Embeddings WHERE ready')
        return [row[0] for row in c]

    def get_url_vectors(
            self, url: str, *,
            embedding_ids: Optional[Iterable[int]] = None,
            raise_ignored: bool = False,
    ) -> Iterable[Tuple[int, Any]]:
        """Computes and yields embedding vector data for the given url."""
        if embedding_ids is None:
            embedding_ids = self.list_ready_embedding_ids()
        for embedding_id in embedding_ids:
            try:
                yield embedding_id, self.get_embedding(embedding_id).transform(url=url)
            except RequestIgnored:
                if raise_ignored:
                    raise
                continue

    def print_embeddings(self, *, w=90):
        c = self.execute(
            'SELECT embedding_id, dim, ready, name, version FROM Embeddings')
        print()
        print()
        print('Embeddings:')
        print()
        c = c.fetchall()
        max_name_len = max(max(map(lambda _row: len(_row[3]), c), default=0), 4)
        print(f" embedding_id\t dim\tstatus   \t{'name'+' '*(max_name_len-4)}\tversion")
        print('─' * w)
        for embedding_id, dim, ready, name, version in c:
            try:
                status: str = {
                    True: '\033[92mREADY    \033[0m',
                }[ready]
            except KeyError:
                status: str = '\033[33mUNKNOWN  \033[0m'
            print(f" {embedding_id:12d}\t{dim:4d}\t{status:s}\t{name+' '*(max_name_len-len(name))}\t{version}")
        print('─' * w)
