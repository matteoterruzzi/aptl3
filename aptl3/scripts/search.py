import logging

from ..db import Database


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Search something using a natural text query')

    parser.add_argument('db', type=str, help='database data directory')
    parser.add_argument('query', type=str, help='natural text query')

    parser.add_argument('--log', default='INFO', type=str, help='logging level (defaults to INFO)')

    args = parser.parse_args()

    logging.basicConfig(level=args.log.upper(),
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S')

    db = Database(data_dir=args.db)
    db.execute('PRAGMA synchronous = OFF')
    results_id, inserted = db.search(args.query, n=100)
    c = db.execute(
        f'SELECT rank, media_id, url '
        f'FROM results.ResultsMediaFiltered NATURAL JOIN MediaLocations '
        f'WHERE results_id = ? ORDER BY rank DESC ', (results_id,))
    for rank, media_id, url in c:
        print(rank, media_id.hex(), url, sep='\t')


if __name__ == '__main__':
    exit(main())
