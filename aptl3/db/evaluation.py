from typing import Optional, Iterable, Tuple, Dict, NamedTuple

from .relations import RelationsDatabase
from .manifolds import ManifoldsDatabase


class AnnEvaluationStats(NamedTuple):
    samples: int
    recall_at_k: Dict[int, float]
    cosine_mean: float
    cosine_std: float
    top_cosine_mean: float
    top_cosine_std: float


class SameSpaceEvaluationDatabase(ManifoldsDatabase, RelationsDatabase):

    def __init__(self, data_dir: Optional[str] = None):
        super().__init__(data_dir)

    def __search_matches(self,
                         x_embedding_id: int,
                         y_embedding_id: int,
                         relation_ids: Iterable[int], *,
                         limit: int, n: int,
                         ) -> Iterable[Tuple[bytes, bytes, int, float, float]]:
        assert n < 1000, n  # for sanity

        c = self.execute(
            f'SELECT REL.media_id, MI_X.manifold_id, MI_X.item_i, '
            f'  REL.other_media_id, MI_Y.manifold_id, MI_Y.item_i, '
            f'  RANDOM() as rand_ '
            f'FROM MediaRelations REL '
            f'JOIN ManifoldItems MI_X ON REL.media_id = MI_X.media_id '
            f'JOIN ManifoldItems MI_Y ON REL.other_media_id = MI_Y.media_id '
            f'JOIN Manifolds M_X ON MI_X.manifold_id = M_X.manifold_id '
            f'JOIN Manifolds M_Y ON MI_Y.manifold_id = M_Y.manifold_id '
            f'WHERE relation_id IN ({", ".join(str(int(_r)) for _r in relation_ids)}) '
            f'AND M_X.embedding_id = {int(x_embedding_id):d} '
            f'AND M_Y.embedding_id = {int(y_embedding_id):d} '
            f'UNION '
            f'SELECT REL.other_media_id, MI_X.manifold_id, MI_X.item_i, '
            f'  REL.media_id, MI_Y.manifold_id, MI_Y.item_i, '
            f'  RANDOM() as rand_ '
            f'FROM MediaRelations REL '
            f'JOIN ManifoldItems MI_X ON REL.other_media_id = MI_X.media_id '
            f'JOIN ManifoldItems MI_Y ON REL.media_id = MI_Y.media_id '
            f'JOIN Manifolds M_X ON MI_X.manifold_id = M_X.manifold_id '
            f'JOIN Manifolds M_Y ON MI_Y.manifold_id = M_Y.manifold_id '
            f'WHERE relation_id IN ({", ".join(str(int(_r)) for _r in relation_ids)}) '
            f'AND M_X.embedding_id = {int(x_embedding_id):d} '
            f'AND M_Y.embedding_id = {int(y_embedding_id):d} '
            f'ORDER BY rand_ LIMIT {int(limit):d}'
        )

        for x_mid, x_manifold_id, x_item_i, y_mid, y_manifold_id, y_item_i, _ in c:
            x = self.get_item_vector(x_manifold_id, x_item_i)
            y = self.get_item_vector(y_manifold_id, y_item_i)
            dis = sum((xi-yi)**2 for xi, yi in zip(x, y))**.5

            # NOTE: finding y_mid across different manifolds by ann
            neighbours = self.find_vector_neighbours(y_embedding_id, x, n=n)
            neighbours = list(neighbours)
            neighbours.sort(key=lambda row_: row_[2])
            neighbours = neighbours[:n]

            if neighbours:
                top_dis = neighbours[0][2]
            else:
                top_dis = 2

            for i, (ann_y_manifold_id, ann_y_item_i, ann_dis) in enumerate(neighbours):
                c2 = self.execute(
                    f'SELECT media_id FROM ManifoldItems '
                    f'WHERE manifold_id = ? AND item_i = ? LIMIT 1',
                    (ann_y_manifold_id, ann_y_item_i))
                row = c2.fetchone()
                if row is None:
                    continue  # The item is in the index, but not in the db
                ann_y_mid: bytes = row[0]
                if ann_y_mid == y_mid:  # exact match found!
                    yield x_mid, y_mid, i, min(ann_dis, dis), top_dis
                    break
            else:
                yield x_mid, y_mid, 1 << 30, dis, top_dis

    def evaluate_same_space(self,
                            x_embedding_id: int,
                            y_embedding_id: int,
                            relation_ids: Iterable[int], *,
                            limit: int = 100000, n: int = 100,
                            kk: Tuple[int] = (1, 5, 10, 100),
                            ) -> AnnEvaluationStats:
        # noinspection PyPackageRequirements
        from tqdm import tqdm

        assert all(n >= k for k in kk), (n, kk)

        samples = 0
        recall_at_k = {k: 0 for k in kk}
        cosine_sum = 0
        cosine2_sum = 0
        top_cosine_sum = 0
        top_cosine2_sum = 0

        matches = self.__search_matches(
            x_embedding_id, y_embedding_id, relation_ids, limit=limit, n=n)

        for x_mid, y_mid, i, dis, top_dis in tqdm(matches):
            samples += 1
            for k in kk:
                recall_at_k[k] += i < k
            cos = 1 - dis ** 2 / 2
            cosine_sum += cos
            cosine2_sum += cos**2
            top_cos = 1 - top_dis ** 2 / 2
            top_cosine_sum += top_cos
            top_cosine2_sum += top_cos**2

        for k in kk:
            recall_at_k[k] /= (samples or 1)
        cosine_mean = cosine_sum / (samples or 1)
        cosine_std = (cosine2_sum / (samples or 1) - cosine_mean ** 2) ** 0.5

        return AnnEvaluationStats(
            samples=samples,
            recall_at_k=recall_at_k,
            cosine_mean=cosine_mean,
            cosine_std=cosine_std,
            top_cosine_mean=cosine_mean,
            top_cosine_std=cosine_std,
        )
