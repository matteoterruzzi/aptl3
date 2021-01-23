import json
from contextlib import nullcontext
from typing import Iterable, Tuple

from .schema import SchemaDatabase


class EquivalenceConsistencyViolation(Exception):
    pass


class RelationsDatabase(SchemaDatabase):

    def create_relation(self, metadata: dict, *,
                        commit: bool = True) -> int:
        with (self._db if commit else nullcontext()):
            c = self._db.execute('INSERT INTO Relations (metadata) VALUES (?)', (json.dumps(metadata),))
            return c.lastrowid

    def add_media_relation(self, relation_id: int, media_id: bytes, other_media_id: bytes, *,
                           commit: bool = True) -> None:
        with (self._db if commit else nullcontext()):
            self._db.execute('INSERT OR IGNORE INTO MediaRelations (relation_id, media_id, other_media_id) '
                             'VALUES (?, ?, ?) ', (relation_id, media_id, other_media_id))

    def add_media_relations(self, relation_id: int, media_id_pairs: Iterable[Tuple[bytes, bytes]], *,
                            commit: bool = True, batch_size: int) -> int:
        def _insert(_params):
            self._db.executemany(
                'INSERT OR IGNORE INTO MediaRelations (relation_id, media_id, other_media_id) '
                'VALUES (?, ?, ?) ', _params)
            return len(_params)

        inserted = 0
        with (self._db if commit else nullcontext()):
            batch = set()
            for ml, mr in media_id_pairs:
                batch.add((relation_id, ml, mr))
                if len(batch) >= batch_size:
                    inserted += _insert(batch)
                    batch.clear()
            if batch:
                inserted += _insert(batch)
        return inserted

    def _drop_relations_indexes(self):
        """This should never be useful"""
        self._db.executescript(
            'DROP INDEX IF EXISTS MediaRelations_media_id;'
            'DROP INDEX IF EXISTS MediaRelations_other_media_id;')

    def _create_relations_indexes(self):
        self._db.executescript(
            'CREATE INDEX IF NOT EXISTS MediaRelations_media_id ON MediaRelations(media_id);'
            'CREATE INDEX IF NOT EXISTS MediaRelations_other_media_id ON MediaRelations(other_media_id);')
