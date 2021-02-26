import logging
import tempfile

# noinspection PyPackageRequirements
from matplotlib import pyplot as plt

from ..db import Database


def full_cycle(self: Database, *, coco_dir: str,
               x_embedding_name: str = 'sentence',
               y_embedding_name: str = 'image',
               ):
    from .load_coco import load_coco
    from .learning_curves import plot_learning_curves

    relation_id_train2017: int = load_coco(
        self, coco_dir=coco_dir, data_type='train2017', max_samples=16384, only_one_caption=True)
    relation_id_val2017: int = load_coco(
        self, coco_dir=coco_dir, data_type='val2017', max_samples=5000, only_one_caption=True)

    x_embedding_id: int = self.get_embedding_id(x_embedding_name)
    y_embedding_id: int = self.get_embedding_id(y_embedding_name)

    self.build_new_manifold(embedding_id=x_embedding_id)
    self.build_new_manifold(embedding_id=y_embedding_id)

    plt.figure()
    plot_learning_curves(
        self=self,
        x_embedding_name=x_embedding_name,
        y_embedding_name=y_embedding_name,
        train_relation_ids=[relation_id_train2017],
        val_relation_ids=[relation_id_val2017],
        samples=[100, 320, 1000, 3200, 10000, 16384],
        val_samples=5000,
        max_query=5000,
        search_n=3200,
    )
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate the quality of the retrieval models.')

    parser.add_argument('coco', type=str, help='COCO dataset directory')
    parser.add_argument('--log', default='INFO', type=str, help='logging level (defaults to INFO)')

    args = parser.parse_args()

    logging.basicConfig(level=args.log.upper(),
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S')

    data_dir = tempfile.mkdtemp(prefix='full_cycle.', dir='.')
    db = Database(data_dir=data_dir)
    db.execute('PRAGMA synchronous = OFF')
    full_cycle(db, coco_dir=args.coco)


if __name__ == '__main__':
    exit(main())
