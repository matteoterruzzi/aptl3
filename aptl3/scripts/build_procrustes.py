import logging

from ..db import Database


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Create a generalized procrustes analysis for cross-modal retrieval')
    parser.add_argument('db', type=str,
                        help='database data directory')
    parser.add_argument('embedding', type=str, nargs='+',
                        help='source embedding model names (specify at least 2)')
    parser.add_argument('--relation', type=int, nargs='+', metavar='ID', required=True,
                        help='identifier numbers of relations to be used to find alignments, '
                             'with reflexive, symmetric (tolerance) closure (specify at least 1)')
    parser.add_argument('--min-samples', default=100, type=int,
                        help='Minimum acceptable training set size (defaults to 100)')
    parser.add_argument('--max-samples', default=10000, type=int,
                        help='Maximum number of unique random samples (defaults to 10000)')
    parser.add_argument('--log', default='INFO', type=str,
                        help='logging level (defaults to INFO)')
    parser.epilog = "If you specify --relation and no other flag, please use the terminator '--'."

    args = parser.parse_args()

    logging.basicConfig(level=args.log.upper(),
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S')

    db = Database(args.db)
    db.build_generalized_procrustes(
        src_embedding_ids=[db.get_embedding_id(en) for en in args.embedding],
        src_relation_ids=args.relation,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
    )


if __name__ == '__main__':
    exit(main())
