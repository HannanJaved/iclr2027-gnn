from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DATASETS = ("cora", "citeseer", "pubmed")
GRAPH_SEEDS = tuple(range(5))
TRAIN_SEEDS = tuple(range(50))
OUTPUT_ROOT = Path("outputs/counterfactual_retraining")
BASELINE_SET_TEMPLATE = "{dataset}-gcn-seed-relative0.1-epochs200"
EPSILON = 0.1
MAX_EPOCHS = 200


@dataclass(frozen=True)
class CounterfactualCondition:
    condition_id: str
    mode: str
    strength: float
    strength_id: str
    tau: float | None = None

    @property
    def rashomon_type_prefix(self) -> str:
        return f"cf_retrain_{self.condition_id}"


CONDITIONS = (
    CounterfactualCondition("random_0p05", "random", 0.05, "0p05"),
    CounterfactualCondition("random_0p25", "random", 0.25, "0p25"),
    CounterfactualCondition("local_tau0p1", "local_entanglement", 0.05, "0p05", tau=0.1),
)


def graph_path_for(dataset: str, condition: CounterfactualCondition, graph_seed: int) -> Path:
    if condition.mode == "random":
        return (
            Path("outputs/repeated_rewiring/random/graphs")
            / f"{dataset}-random-{condition.strength_id}-seed{graph_seed}.pt"
        )
    if condition.mode == "local_entanglement":
        tau_id = str(condition.tau).replace(".", "p")
        return (
            Path("outputs/repeated_rewiring/local/graphs")
            / f"{dataset}-local_entanglement-tau{tau_id}-{condition.strength_id}-seed{graph_seed}.pt"
        )
    raise ValueError(f"Unsupported mode: {condition.mode}")


def metadata_path_for(dataset: str, condition: CounterfactualCondition, graph_seed: int) -> Path:
    return graph_path_for(dataset, condition, graph_seed).with_suffix(".json")


def rashomon_type_for(condition: CounterfactualCondition, graph_seed: int) -> str:
    return f"{condition.rashomon_type_prefix}_gseed{graph_seed}"


def set_type_for(condition: CounterfactualCondition, graph_seed: int) -> str:
    return rashomon_type_for(condition, graph_seed)


def decode_array_task(task_id: int) -> tuple[str, CounterfactualCondition, int, int]:
    """Decode flat array index over datasets × conditions × graph seeds × train seeds."""
    n_train = len(TRAIN_SEEDS)
    n_graph = len(GRAPH_SEEDS)
    n_cond = len(CONDITIONS)
    n_datasets = len(DATASETS)
    total = n_datasets * n_cond * n_graph * n_train
    if not 0 <= task_id < total:
        raise ValueError(f"task_id={task_id} outside [0, {total}).")
    train_seed = TRAIN_SEEDS[task_id % n_train]
    rest = task_id // n_train
    graph_seed = GRAPH_SEEDS[rest % n_graph]
    rest //= n_graph
    condition = CONDITIONS[rest % n_cond]
    dataset = DATASETS[rest // n_cond]
    return dataset, condition, graph_seed, train_seed


def expected_job_count() -> int:
    return len(DATASETS) * len(CONDITIONS) * len(GRAPH_SEEDS) * len(TRAIN_SEEDS)
