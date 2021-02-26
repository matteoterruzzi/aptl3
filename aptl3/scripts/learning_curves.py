import logging
import pickle
import tempfile
from typing import List

import numpy as np
# noinspection PyPackageRequirements
from matplotlib import pyplot as plt

from ..db import Database


def plot_learning_curves(
        self: Database,
        x_embedding_name: str,
        y_embedding_name: str,
        train_relation_ids: List[int],
        val_relation_ids: List[int],
        samples: List[int],
        val_samples: int,
        max_query: int,
        search_n: int,
        search_k: int = -1,
        ):

    x_embedding_id = self.get_embedding_id(x_embedding_name)
    y_embedding_id = self.get_embedding_id(y_embedding_name)

    src_embedding_ids = [x_embedding_id, y_embedding_id]

    c = self.execute(
        'SELECT COUNT(DISTINCT media_id) '
        'FROM ManifoldItems NATURAL JOIN Manifolds M '
        'JOIN Embeddings E on M.embedding_id = E.embedding_id '
        'WHERE E.embedding_id = ? ', (y_embedding_id,))
    _search_space: int = c.fetchone()[0]
    logging.info(f'{_search_space=:d}')

    _learning_curves = np.empty(shape=[len(samples), 2], dtype=np.float32)

    for _samples_i, _samples in enumerate(samples):

        gpa_id, gpa = self.build_generalized_procrustes(
            src_embedding_ids=src_embedding_ids, src_relation_ids=train_relation_ids,
            min_samples=_samples, max_samples=_samples, keep_training_data=True)

        logging.info(f'Evaluating {gpa_id=:d} on training set...')
        out = self.evaluate_generalized_procrustes(
            gpa=gpa, src_relation_ids=None,
            min_samples=_samples, max_samples=_samples, max_query=max_query,
            search_n=search_n, search_k=search_k,
            embedding_id_pairs=[(x_embedding_id, y_embedding_id)]
        )

        for xe, ye, train_ranks, train_top_dis, train_true_dis in out:
            assert (xe, ye) == (x_embedding_id, y_embedding_id)
            _learning_curves[_samples_i, 0] = np.mean(train_true_dis)
            break
        else:
            raise RuntimeError("No result from evaluate_generalized_procrustes")

        logging.info(f'Evaluating {gpa_id=:d} on validation set...')
        self.forget_generalized_procrustes_training_data()

        out = self.evaluate_generalized_procrustes(
            gpa=gpa, src_relation_ids=val_relation_ids,
            min_samples=val_samples, max_samples=val_samples, max_query=max_query,
            search_n=search_n, search_k=search_k,
            embedding_id_pairs=[(x_embedding_id, y_embedding_id)]
        )

        for xe, ye, val_ranks, val_top_dis, val_true_dis in out:
            assert (xe, ye) == (x_embedding_id, y_embedding_id)
            _learning_curves[_samples_i, 1] = np.mean(val_true_dis)
            break
        else:
            raise RuntimeError("No result from evaluate_generalized_procrustes")

    try:
        with tempfile.NamedTemporaryFile(
                dir=self.get_data_dir(), prefix='learning_curves.', suffix='.pickle', delete=False) as f:
            stuff = {
                'x_embedding_name': x_embedding_name,
                'y_embedding_name': y_embedding_name,
                'train_relation_ids': train_relation_ids,
                'val_relation_ids': val_relation_ids,
                'samples': samples,
                'val_samples': val_samples,
                'max_query': max_query,
                'search_n': search_n,
                'search_k': search_k,
                'learning_curves': _learning_curves,
            }
            pickle.dump(stuff, f, protocol=5)
            logging.info(f'Results data pickled to file {f.name}')
    except:
        pass

    plt.title(
        f'Learning curves of GPA model \n'
        f'find {y_embedding_name} by {x_embedding_name} \n'
        f'validation set samples: {val_samples:d} \n'
        f'search space: {_search_space:d} items')

    plt.xlabel('Training samples')
    plt.xscale('log')
    plt.ylabel('Avg retrieval angular distance [deg]')

    for _set_i, _set in enumerate(['train', 'val']):
        plt.plot(samples, np.rad2deg(np.arccos(1 - _learning_curves[:, _set_i]**2 / 2)), label=f'{_set:s}')

    plt.ylim((0., None))
    plt.grid('both', 'both')
    plt.legend()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate the quality of the retrieval models.')
    parser.add_argument('db', type=str, help='database data directory')

    parser.add_argument('x_embedding', type=str, help='source embedding model name (e.g. sentence)')
    parser.add_argument('y_embedding', type=str, help='destination embedding model name (e.g. image)')

    parser.add_argument('--train-relation', type=int, nargs='+', metavar='ID', required=True,
                        help='identifier numbers of relations to be used to find alignments for training, '
                             'with reflexive, symmetric (tolerance) closure (specify at least 1)')
    parser.add_argument('--val-relation', type=int, nargs='+', metavar='ID', required=True,
                        help='identifier numbers of relations to be used to find alignments for validation, '
                             'with reflexive, symmetric (tolerance) closure (specify at least 1)')

    parser.add_argument('--log', default='INFO', type=str, help='logging level (defaults to INFO)')

    args = parser.parse_args()

    logging.basicConfig(level=args.log.upper(),
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S')

    db = Database(args.db)
    db.execute('PRAGMA synchronous = OFF')

    plt.figure()
    plot_learning_curves(
        self=db,
        x_embedding_name=args.x_embedding,
        y_embedding_name=args.y_embedding,
        train_relation_ids=args.train_relation,
        val_relation_ids=args.val_relation,
        samples=[100, 320, 1000, 3200, 10000, 16384, 32000, 100000],
        val_samples=5000,
        max_query=500000,
        search_n=3200,
    )
    plt.show()


if __name__ == '__main__':
    exit(main())

