import numpy as np

from ...embedding.sentence import SentenceEmbedding


def test_sentence_shape_norm_dis():
    _urls = ['data:,' + _sentence for _sentence in [
        'This is a sentence for the test case.',
        'The present one is a test sentence.',  # This is more similar to the first ...
        'The quick brown fox jumps over the lazy dog.',  # ... than this one.
    ]]

    e = SentenceEmbedding()

    vectors = []
    for _url in _urls:
        v = e.transform(url=_url)
        assert tuple(v.shape) == (e.get_dim(),)
        assert abs(float(np.linalg.norm(v, ord=2, axis=None, keepdims=False) - 1.)) < 1.0e-5
        vectors.append(v)

    dis_1 = np.linalg.norm(vectors[0] - vectors[1], ord=2, axis=None, keepdims=False)
    dis_2 = np.linalg.norm(vectors[0] - vectors[2], ord=2, axis=None, keepdims=False)

    assert dis_1 < dis_2
    assert dis_1 < 0.9
    assert dis_2 > 0.2
