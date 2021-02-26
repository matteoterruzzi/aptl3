import logging
from time import sleep

from ..db.thumbnails import ThumbnailsDatabase


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Load missing thumbnails')
    parser.add_argument('db', type=str, help='database data directory')
    parser.add_argument('--log', default='INFO', type=str, help='logging level (defaults to INFO)')

    args = parser.parse_args()

    logging.basicConfig(level=args.log.upper(),
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S')

    db = ThumbnailsDatabase(args.db)
    db.thumbs_load()
    n = 3
    for i in range(0, n):
        try:
            while db.thumbs_process.pending:
                print(db.thumbs_process.status, end='  \r')
                sleep(2)
        except KeyboardInterrupt:
            print()
            db.thumbs_finish()
            logging.info(f'{db.thumbs_progress} {db.thumbs_process.status}')
            logging.info(f'Interrupt {i+1}/{n}.' +
                         ('Finishing...' if i+1 < n else 'Stopping!'))
    db.thumbs_process.stop()
    logging.info(db.thumbs_progress)


if __name__ == '__main__':
    exit(main())
