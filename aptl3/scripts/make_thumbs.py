import logging

from ..db import Database


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Load missing thumbnails')
    parser.add_argument('db', type=str, help='database data directory')
    parser.add_argument('--log', default='INFO', type=str, help='logging level (defaults to INFO)')

    args = parser.parse_args()

    logging.basicConfig(level=args.log.upper(),
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')

    missing, inserted = Database(args.db).try_load_missing_thumbs()
    finished = missing == 0
    print()
    print(f'{finished=} {inserted=}')


if __name__ == '__main__':
    exit(main())
