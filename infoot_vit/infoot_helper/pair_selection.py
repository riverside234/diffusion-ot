"""Fit-time image-pair selection; inference only consumes the saved edge set."""
import torch


def select_pairs(router, source_ids, target_ids, top_k, *, chunk_size=16):
    k = len(target_ids) if top_k is None else min(top_k, len(target_ids))
    stable_ids = sorted(range(len(target_ids)),key=lambda j:target_ids[j])
    rows = []
    for start in range(0, len(source_ids), chunk_size):
        probabilities = router.conditional_weights(router.source[start:start + chunk_size])
        for offset, row in enumerate(probabilities):
            order = [stable_ids[j] for j in torch.argsort(row[stable_ids],descending=True,stable=True)[:k].tolist()]
            retained = float(row[order].sum())
            rows.append(dict(source_id=source_ids[start + offset], target_ids=[target_ids[j] for j in order],
                probabilities=row[order].tolist(), retained_probability=retained,
                discarded_probability=max(0., 1. - retained)))
    return dict(schema="infoot_pair_selection_v1", fit_pair_top_k=top_k, effective_k=k,
        rule="router_conditional_probability_descending_then_stable_target_id",
        source_ids=source_ids, target_ids=target_ids, rows=rows, pair_count=len(source_ids) * k)


def selection_edges(selection, source_ids, target_ids, top_k):
    k = len(target_ids) if top_k is None else min(top_k, len(target_ids))
    if (selection.get("schema") != "infoot_pair_selection_v1" or selection["source_ids"] != source_ids
            or selection["target_ids"] != target_ids or selection["fit_pair_top_k"] != top_k
            or selection["effective_k"] != k or len(selection["rows"]) != len(source_ids)
            or selection["pair_count"] != len(source_ids) * k):
        raise ValueError("Pair selection does not match the fitted support/configuration.")
    edges = set()
    for sid, row in zip(source_ids, selection["rows"]):
        ids = row["target_ids"]
        probabilities = torch.tensor(row["probabilities"], dtype=torch.float64)
        if (row["source_id"] != sid or len(ids) != k or len(set(ids)) != k or not set(ids) <= set(target_ids)
                or len(probabilities) != k or not torch.isfinite(probabilities).all()
                or (probabilities < 0).any() or float(probabilities.sum()) <= 0
                or float(probabilities.sum()) > 1 + 1e-12
                or abs(float(probabilities.sum()) - row["retained_probability"]) > 1e-12
                or abs(1 - row["retained_probability"] - row["discarded_probability"]) > 1e-12):
            raise ValueError("Invalid persisted image-pair selection row.")
        edges.update((sid, tid) for tid in ids)
    return edges
