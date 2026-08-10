import numpy as np

from gnn_rashomon.multiplicity.predictive_entropy import predictive_entropy
from gnn_rashomon.multiplicity.rashomon_capacity import probability_diameter, rashomon_capacity
from gnn_rashomon.multiplicity.variation_ratio import variation_ratio


def test_predictive_entropy_known_values():
    probabilities = np.array(
        [
            [[1.0, 0.0], [0.5, 0.5]],
            [[1.0, 0.0], [0.5, 0.5]],
        ]
    )
    entropy = predictive_entropy(probabilities)
    assert np.allclose(entropy[0], 0.0, atol=1e-10)
    assert np.allclose(entropy[1], np.log(2.0), atol=1e-10)


def test_variation_ratio_known_values():
    probabilities = np.array(
        [
            [[0.9, 0.1], [0.9, 0.1]],
            [[0.8, 0.2], [0.1, 0.9]],
            [[0.7, 0.3], [0.2, 0.8]],
        ]
    )
    assert np.allclose(variation_ratio(probabilities), [0.0, 1.0 / 3.0])


def test_probability_diameter_known_values():
    probabilities = np.array(
        [
            [[1.0, 0.0], [0.6, 0.4]],
            [[0.0, 1.0], [0.2, 0.8]],
        ]
    )
    assert np.allclose(probability_diameter(probabilities), [1.0, 0.4])
    assert np.allclose(rashomon_capacity(probabilities), [1.0, 0.4])


def test_probability_diameter_chunked_matches_exact():
    probabilities = np.array(
        [
            [[1.0, 0.0], [0.6, 0.4]],
            [[0.0, 1.0], [0.2, 0.8]],
            [[0.5, 0.5], [0.8, 0.2]],
        ]
    )
    assert np.allclose(
        probability_diameter(probabilities, method="chunked", chunk_size=1),
        probability_diameter(probabilities, method="exact"),
    )
