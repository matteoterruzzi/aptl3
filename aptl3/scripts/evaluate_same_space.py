import logging

from ..db.evaluation import SameSpaceEvaluationDatabase


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate the quality of same-space cross-modal retrieval.')

    parser.add_argument('db', type=str, help='database data directory')
    parser.add_argument('x_embedding', type=str, help='source embedding model name')
    parser.add_argument('y_embedding', type=str, help='destination embedding model name')
    parser.add_argument('--relation', type=int, nargs='+', metavar='ID', required=True,
                        help='identifier numbers of relations to be used to find alignments '
                             '(specify at least 1), e.g. COCO training or validation set')

    parser.add_argument('--log', default='INFO', type=str, help='logging level (defaults to INFO)')

    args = parser.parse_args()

    logging.basicConfig(level=args.log.upper(),
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S')

    db = SameSpaceEvaluationDatabase(data_dir=args.db)
    db.execute('PRAGMA synchronous = OFF')
    stats = db.evaluate_same_space(
        x_embedding_id=db.get_embedding_id(args.x_embedding),
        y_embedding_id=db.get_embedding_id(args.y_embedding),
        relation_ids=tuple(args.relation))
    logging.info('DONE.')
    print(stats)


if __name__ == '__main__':
    exit(main())
