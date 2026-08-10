from gnn_rashomon.rashomon.filtering import filter_by_training_loss, validate_partition
from gnn_rashomon.rashomon.tolerance import is_retained, loss_threshold


def test_absolute_tolerance():
    assert loss_threshold(1.0, "absolute", 0.1) == 1.1
    assert is_retained(1.1, 1.0, "absolute", 0.1)
    assert not is_retained(1.11, 1.0, "absolute", 0.1)


def test_relative_tolerance():
    assert loss_threshold(2.0, "relative", 0.1) == 2.2
    assert is_retained(2.2, 2.0, "relative", 0.1)
    assert not is_retained(2.21, 2.0, "relative", 0.1)


def test_filter_partition():
    losses = {"a": 1.0, "b": 1.05, "c": 1.2}
    retained, rejected = filter_by_training_loss(losses, 1.0, "absolute", 0.05)
    assert retained == ["a", "b"]
    assert rejected == ["c"]
    validate_partition(losses.keys(), retained, rejected)
