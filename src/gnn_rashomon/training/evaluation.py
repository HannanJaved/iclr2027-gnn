from __future__ import annotations


def masked_loss_and_accuracy(logits: object, labels: object, mask: object) -> tuple[float, float]:
    import torch

    masked_logits = logits[mask]
    masked_labels = labels[mask]
    loss = torch.nn.functional.cross_entropy(masked_logits, masked_labels)
    predictions = masked_logits.argmax(dim=-1)
    accuracy = (predictions == masked_labels).float().mean()
    return float(loss.item()), float(accuracy.item())
