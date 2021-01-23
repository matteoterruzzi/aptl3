import logging
from datetime import datetime
from threading import Thread, Event

from pycocotools.coco import COCO

from ..db import Database

logger = logging.getLogger('load_coco')


def load_coco(db: Database, *, data_type: str = 'train2017', max_samples: int):
    try:
        metadata = dict(
            source='MSCOCO',
            creation=str(datetime.now()),
            data_type=data_type,
        )

        coco = COCO(f'./data/coco_annotations/captions_{data_type}.json')
        total_len = len(coco.imgs)

        db.begin_exclusive_transaction()
        relation_id: int = db.create_relation(metadata, commit=False)
        logger.info(f'New relation #{relation_id:d}: {metadata}')
    except KeyboardInterrupt:
        logger.info('Interrupted before starting.')
        return

    stopped = Event()

    def _download(shard, n):
        for i, id_ in zip(range(max_samples), coco.getImgIds()):
            if i % n != shard:
                continue
            img_url = coco.imgs[id_]['coco_url']
            db.fetch(img_url)
            if stopped.is_set():
                break

    shards = 4
    download_ths = [
        Thread(target=_download, args=(_s, shards), daemon=True, name=f'COCO-{_s:d}-of-{shards:d}')
        for _s in range(shards)
    ]
    for th in download_ths:
        th.start()

    def _gen_pairs():
        for i, id_ in zip(range(max_samples), coco.getImgIds()):
            img_url = coco.imgs[id_]['coco_url']
            img_media_id, _flags = db.ingest_url(img_url, commit=False, update_last_access=False)

            for ann_id in coco.getAnnIds(id_):
                ann_url = 'data:,' + coco.anns[ann_id]['caption']
                ann_media_id, _flags = db.ingest_url(ann_url, commit=False, update_last_access=False)

                yield img_media_id, ann_media_id
                print(f"Loading MS COCO ({i}/{total_len} images)...", end="\r", flush=True)

    try:
        db.add_media_relations(relation_id, media_id_pairs=_gen_pairs(), commit=False, batch_size=64)
    except KeyboardInterrupt:
        logger.info(f'Interrupted.')
    db.commit()

    c = db.execute('SELECT COUNT(*) FROM MediaRelations WHERE relation_id = ?', (relation_id,))
    inserted: int = c.fetchone()[0]

    logger.info(f'Added {inserted:d} COCO annotations to relation #{relation_id}.')
    logger.debug(f'{[th.is_alive() for th in download_ths]=}')

    stopped.set()
    for th in download_ths:
        th.join()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Load MS COCO dataset and equivalence')
    parser.add_argument('db', type=str, help='database data directory')
    parser.add_argument('data', type=str, help='dataset subset (train2017 or val2017)')

    parser.add_argument('--limit', default=999999, type=int, help='limit the number of samples (defaults to 999999)')
    parser.add_argument('--log', default='INFO', type=str, help='logging level (defaults to INFO)')

    args = parser.parse_args()

    logging.basicConfig(level=args.log.upper(),
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')

    db = Database(args.db)
    db.execute('PRAGMA synchronous = OFF')
    load_coco(db=db, data_type=args.data, max_samples=args.limit)


if __name__ == '__main__':
    main()
