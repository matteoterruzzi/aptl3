import datetime
import logging
from contextlib import nullcontext
from functools import lru_cache
from typing import Optional, Dict, Tuple, Any, Iterable

from .schema import SchemaDatabase
from ..embedding.abc import Embedding, RequestIgnored


# TODO: add tools to compute embeddings in a separate process, exploiting their static nature, or passing the data dir.


class EmbeddingsDatabase(SchemaDatabase):
    """
    Static naming of embedding models and versioning (or future support for complex configurations)
    """

    def __init__(self, data_dir: Optional[str] = None):
        super().__init__(data_dir)
        self.__logger = logging.getLogger('manifolds')
        # will map embedding id to name and instance
        self.__embedding_names: Dict[int, str] = dict()
        self.__embedding_instances: Dict[int, Embedding] = dict()

    @classmethod
    def initialize_static_embeddings(cls):
        for name in [
            'image',
            'sentence',
        ]:
            cls.make_static_embedding_by_name(name)

    @classmethod
    @lru_cache(maxsize=None)
    def make_static_embedding_by_name(cls, name: str) -> Embedding:
        if name.startswith('r-'):
            from ..embedding.random import NonLSHEmbedding
            dim = int(name[2:])
            return NonLSHEmbedding(dim)
        elif name == 'sentence':
            from ..embedding.sentence import SentenceEmbedding
            return SentenceEmbedding()
        elif name == 'image':
            from ..embedding.image import ImageEmbedding
            return ImageEmbedding()
        else:
            raise KeyError(f'No embedding runtime available for {name}.')

    def get_embedding(self, embedding_id: int) -> Embedding:
        try:
            return self.__embedding_instances[embedding_id]
        except KeyError:
            try:
                name = self.__embedding_names[embedding_id]
            except KeyError:
                c = self.execute('SELECT name FROM Embeddings WHERE embedding_id = ?', (embedding_id,))
                row = c.fetchone()
                if row is None:
                    raise ValueError(f'Embedding #{embedding_id} not found in db.')
                name: str = row[0]
                self.__embedding_names[embedding_id] = name
            embedding = self.make_static_embedding_by_name(name)
            self.__embedding_instances[embedding_id] = embedding
            return embedding

    def get_embedding_id(self, name: str) -> int:
        for _embedding_id, _name in self.__embedding_names.items():
            if _name == name:
                return _embedding_id

        c = self.execute('SELECT embedding_id FROM Embeddings WHERE name = ?', (name,))
        row = c.fetchone()
        if row is None:
            raise ValueError(name)  # not found in db
        else:
            return row[0]

    def push_new_embedding_version(self, name: str, *, commit: bool = True) -> int:
        with (self._db if commit else nullcontext()):
            c = self.execute('SELECT embedding_id FROM Embeddings WHERE name = ?', (name,))
            row = c.fetchone()
            if row is None:
                raise ValueError(f'No embedding for {name}')
            old_embedding_id: int = row[0]
            deprecation_name: str = f'{name} DEPRECATED @ UTC {datetime.datetime.utcnow()}'
            self.execute('UPDATE Embeddings SET name = :name, ready = 0 '
                         'WHERE embedding_id = :embedding_id',
                         {'embedding_id': old_embedding_id,
                          'name': deprecation_name})
            self.__embedding_names[old_embedding_id] = deprecation_name

            return self.add_embedding(name, commit=False)

    def add_embedding(self, name: str, *, commit: bool = True) -> int:
        new_embedding: Embedding = self.make_static_embedding_by_name(name)
        with (self._db if commit else nullcontext()):
            c = self.execute('INSERT INTO Embeddings (dim, name, ready) '
                             'VALUES (?, ?, 1)', (new_embedding.get_dim(), name))
            new_embedding_id: int = c.lastrowid

            self.__embedding_names[new_embedding_id] = name
            self.__embedding_instances[new_embedding_id] = new_embedding

            return new_embedding_id

    def get_url_vectors(
            self, url: str, *,
            embedding_ids: Optional[Iterable[int]] = None,
            raise_ignored: bool = False,
    ) -> Iterable[Tuple[int, Any]]:
        """Computes and yields embedding vector data for the given url."""
        if embedding_ids is None:
            c = self.execute('SELECT embedding_id FROM Embeddings WHERE ready')
            embedding_ids = [row[0] for row in c]
        for embedding_id in embedding_ids:
            try:
                yield embedding_id, self.get_embedding(embedding_id).transform(url=url)
            except RequestIgnored:
                if raise_ignored:
                    raise
                continue
