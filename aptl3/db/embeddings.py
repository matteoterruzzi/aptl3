import datetime
import logging
from contextlib import nullcontext
from functools import lru_cache
from typing import Optional, Dict, Tuple, Any, Iterable

from .schema import SchemaDatabase
from ..embedding.abc import Embedding, RequestIgnored


# TODO: add tools to compute embeddings in a separate process, exploiting their static nature, or passing the data dir.


class DeprecatedEmbeddingVersionException(Exception):
    """This exception is managed by EmbeddingsDatabase.get_embedding_id(name)"""
    pass


class EmbeddingsDatabase(SchemaDatabase):
    """
    Static naming of embedding models and versioning (or future support for complex configurations)
    """

    def __init__(self, data_dir: Optional[str] = None):
        super().__init__(data_dir)
        self.__logger = logging.getLogger('manifolds')
        # will map embedding id to name and instance
        self.__embedding_names_versions: Dict[int, Tuple[str, str]] = dict()
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
                name, version = self.__embedding_names_versions[embedding_id]
            except KeyError:
                c = self.execute('SELECT name, version FROM Embeddings WHERE embedding_id = ?', (embedding_id,))
                row = c.fetchone()
                if row is None:
                    raise ValueError(f'Embedding #{embedding_id} not found in db.')
                name: str = row[0]
                version: str = row[1]
                self.__embedding_names_versions[embedding_id] = name, version
            embedding = self.make_static_embedding_by_name(name)
            if embedding.get_version() != version:
                raise DeprecatedEmbeddingVersionException(f"The model {version:s} has been deprecated.")
            self.__embedding_instances[embedding_id] = embedding
            return embedding

    def get_embedding_id(self, name: str, *, commit: bool = True) -> int:
        for _embedding_id, (_name, _version) in self.__embedding_names_versions.items():
            if _name == name:
                return _embedding_id

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
            c = self.execute('SELECT embedding_id, version FROM Embeddings WHERE name = ?', (name,))
            row = c.fetchone()
            if row is None:
                raise ValueError(f'No embedding for {name}')
            old_embedding_id: int = row[0]
            version: str = row[1]
            deprecation_name: str = f'DEPRECATED-{old_embedding_id:d}-{name}'
            self.execute('UPDATE Embeddings SET name = :name, ready = 0 '
                         'WHERE embedding_id = :embedding_id',
                         {'embedding_id': old_embedding_id,
                          'name': deprecation_name})
            self.__embedding_names_versions[old_embedding_id] = deprecation_name, version

            return self.add_embedding(name, commit=False)

    def add_embedding(self, name: str, *, commit: bool = True) -> int:
        new_embedding: Embedding = self.make_static_embedding_by_name(name)
        version = new_embedding.get_version()
        with (self._db if commit else nullcontext()):
            c = self.execute('INSERT INTO Embeddings (dim, name, version, ready) '
                             'VALUES (?, ?, ?, 1)', (new_embedding.get_dim(), name, version))
            new_embedding_id: int = c.lastrowid

            self.__embedding_names_versions[new_embedding_id] = name, version
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
