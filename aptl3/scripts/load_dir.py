import logging

from ..db import Database


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Load a file directory')
    parser.add_argument('db', type=str, help='database data directory')
    parser.add_argument('dir', type=str, help='file directory to load')

    parser.add_argument('--log', default='INFO', type=str, help='logging level (defaults to INFO)')

    # noinspection DuplicatedCode
    args = parser.parse_args()

    logging.basicConfig(level=args.log.upper(),
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S')

    db = Database(args.db)
    db.execute('PRAGMA synchronous = OFF')
    out = db.ingest_file_directory(args.dir)
    logging.info(f'{out}')


if __name__ == '__main__':
    main()
