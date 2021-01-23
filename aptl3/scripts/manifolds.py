import logging

from ..db.manifolds import ManifoldsDatabase


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Manifolds and approximate nearest neighbors indexes')
    parser.add_argument('db', type=str, help='database data directory')

    parser_commands = parser.add_subparsers(title='command', description='Operations on the manifolds',
                                            required=True, metavar='command')

    commands = [
        ('list', 'List existing manifolds'),
        ('new', 'Compute embedding vector data and build new manifold index'),
        ('merge', 'Merge existing indexes for given embedding model name'),
        ('reindex', 'Just reindex an existing manifold')]

    for cmd, h in commands:

        parser_cmd = parser_commands.add_parser(cmd, help=h)
        parser_cmd.set_defaults(cmd=cmd)

        if cmd in ['new', 'merge']:
            parser_cmd.add_argument('embedding', type=str, help='embedding model name')

        if cmd in ['reindex']:
            parser_cmd.add_argument('manifold_id', type=int, help='manifold numeric identifier')

        if cmd in ['new', 'merge', 'reindex']:
            parser_indexing = parser_cmd.add_argument_group('indexing', 'indexing options')

            parser_indexing.add_argument(
                '--metric', default='euclidean', type=str,
                help='metric name (defaults to euclidean)')
            parser_indexing.add_argument(
                '--trees', default=10, type=int,
                help='forest size: more trees gives higher precision when querying (defaults to 10)')

    parser.add_argument('--log', metavar='LEVEL', default='INFO', type=str, help='logging level (defaults to INFO)')

    args = parser.parse_args()

    logging.basicConfig(level=args.log.upper(),
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')

    db = ManifoldsDatabase(args.db)

    if args.cmd == 'list':
        db.print_manifolds()
    elif args.cmd == 'new':
        db.build_new_manifold(
            embedding_id=db.get_embedding_id(args.embedding),
            metric=args.metric,
            n_trees=args.trees,
        )
    elif args.cmd == 'merge':
        db.merge_manifolds(
            embedding_id=db.get_embedding_id(args.embedding),
            metric=args.metric,
            n_trees=args.trees,
        )
    elif args.cmd == 'reindex':
        db.reindex_manifold(
            manifold_id=args.manifold_id,
            metric=args.metric,
            n_trees=args.trees,
        )
    else:
        raise RuntimeError


if __name__ == '__main__':
    exit(main())
