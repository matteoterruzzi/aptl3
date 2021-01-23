import logging
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
from typing import Tuple, Optional

from .locations import LocationsDatabase
from ..images import getSmallImage, imageToBytesIO


def _url_to_thumb_bytes(url: str) -> Optional[bytes]:
    try:
        if url.startswith('data:,'):
            return None
        im = getSmallImage(url)
        thmb = imageToBytesIO(im)
    except Exception:
        logging.debug(f'Skipping {url}', exc_info=True)
        return None
    else:
        return thmb.getvalue()
    finally:
        print('#', end='', flush=True)


class ThumbnailsDatabase(LocationsDatabase):
    def try_load_missing_thumbs(self) -> Tuple[int, int]:
        # TODO: Add option to retry failed

        c = self._db.execute('SELECT COUNT(*) FROM MediaLocations '
                             'WHERE media_id not in (select media_id from Thumbnails) '
                             'AND url NOT LIKE \'data:,%\' '
                             'AND media_id is not NULL')
        logging.info(f'Will try to compute {c.fetchone()[0]:d} missing thumbnails...')

        # NOTE: the thumbnail extractor is very strange. This will be faster with more processes than the number of CPUs
        max_workers = 24
        chunksize = 4
        ppe = ProcessPoolExecutor(max_workers=max_workers)
        logging.info(f'Processing thumbnails in batches of {chunksize*max_workers=:d} with {max_workers=:d}...')

        start_time = perf_counter()
        inserted = 0
        limit = chunksize * max_workers  # NOTE: this will also determine the length of a progress bar
        while limit != 0:
            rate = inserted / (perf_counter() - start_time)
            with self._db:
                c = self._db.execute('SELECT media_id, url FROM MediaLocations '
                                     'WHERE media_id not in (select media_id from Thumbnails) '
                                     'AND url NOT LIKE \'data:,%\' '
                                     'AND media_id is not NULL ORDER BY ts_last_access LIMIT ?', (limit,))
                # TODO: take the batch by inserting a row with null value,
                #  to avoid duplicated computation from other processes.

                try:
                    batch = list(c)
                    print("\r" + ' '*limit + '                                  ', end="\r")
                    print("\r" + '.'*len(batch) + f" {inserted=:d} {rate=:.2f}/s  ", end="\r", flush=True)
                    thumbs = [(m, t) for (m, url), t in zip(
                        batch, ppe.map(_url_to_thumb_bytes, [url for m, url in batch], chunksize=chunksize))]
                except KeyboardInterrupt:
                    print()
                    logging.info(f"Interrupted.")
                    return limit, inserted

                limit = len(thumbs)
                if limit > 0:
                    start_time += self.begin_exclusive_transaction()
                    c = self._db.executemany('INSERT OR IGNORE INTO Thumbnails (media_id, thumbnail) '
                                             'VALUES (?, ?)', thumbs)
                    self._db.commit()
                    inserted += len(thumbs)

        ppe.shutdown()

        # TODO: implement a pipeline similar to the one for manifolds, also to experiment with multiprocessing.

        return limit, inserted
