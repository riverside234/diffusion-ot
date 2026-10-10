"""Reproducible image sampling, before any image/patch transport is fitted."""
import random


def sample_ids(ids, count=None, seed=42):
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("Sampling requires unique nonempty source IDs.")
    if type(seed) is not int or (count is not None and (type(count) is not int or count < 1)):
        raise ValueError("Sample count must be positive (or null for all); seed must be an integer.")
    if count is not None and count > len(ids):
        raise ValueError(f"Requested {count} training images, but only {len(ids)} available; no replacement or silent reduction.")
    if count is None:
        return list(ids)
    # Canonicalize the population before drawing, and the selected order after.
    return sorted(random.Random(seed).sample(sorted(ids), count))


def sampling_record(ids, selected, seed):
    return dict(algorithm="python_random_sample_sorted_ids_v1", seed=seed, replacement=False,
                population_count=len(ids), selected_count=len(selected), ordered_ids=selected)
