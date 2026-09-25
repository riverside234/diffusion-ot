"""Config-aware Stage 1B JSONL presentation; never changes training tensors.

Schema 2 keeps active loss names compatible with old logs; the PDAE-only
experiment uses schema 3 with no legacy teacher/critic fields. Diagnostic-only
image distances live under ``diagnostics`` instead of masquerading as losses.
Filtering uses configured weights, never measured values: an enabled loss
that reaches zero (or has a zero warmup coefficient) must remain visible.
"""
from __future__ import annotations

from copy import deepcopy


def _positive(options, key, default=0.):
    return float(options.get(key, default)) > 0


class Stage1BLogFormatter:
    def __init__(self, config):
        weights = config.get("loss_weights") or {}
        generator = config.get("generator_adaptation") or {}
        image = config.get("decoded_translation") or {}
        protection = config.get("matching_regularization") or {}
        contrast = config.get("matching_contrastive") or {}
        self.decoder = bool(generator.get("enabled", False) and image.get("enabled", False))
        self.self_supervised = image.get("supervision", "external") == "self_supervised"
        self.patchnce = self.self_supervised and image.get("objective", "source_infonce") == "patchnce"
        self.enabled = {
            **{f"{d}_reconstruction": _positive(weights, f"{d}_reconstruction", 1.) for d in ("cat", "dog")},
            "infoot_alignment": _positive(weights, "infoot_alignment", .02)
                and _positive(config.get("infoot") or {}, "mi_weight", 1.),
            "latent_anchor": _positive(weights, "latent_anchor", .01),
            "semantic_neighborhood": _positive(weights, "semantic_neighborhood"),
            "conditional_structure": bool((config.get("conditional_structure") or {}).get("enabled", False))
                and _positive(weights, "conditional_structure"),
            "projection_support": bool((config.get("projection_support") or {}).get("enabled", False))
                and _positive(weights, "projection_support"),
            **{f"matching_{part}": bool(protection.get("enabled", False)) and _positive(protection, f"{part}_weight", default)
               for part, default in (("variance", .02), ("covariance", .001))},
            **{f"matching_{part}_contrastive": bool(contrast.get("enabled", False)) and _positive(contrast, f"{part}_weight", .01)
               for part in ("neighborhood", "conditional")},
            **{part: self.decoder and _positive(generator, f"{part}_weight", default)
               for part, default in (("null_preservation", .1), ("conditioned_preservation", 0.))},
            **{part: self.decoder and _positive(image, f"{part}_weight", default)
               for part, default in (("structure", .1), ("structure_contrastive", 0.),
                                     ("perceptual", 0.), ("adversarial", .01), ("code_consistency", 0.))},
            "color_histogram": self.decoder and _positive(image.get("color_histogram") or {}, "weight"),
            "source_contrastive": self.decoder and self.self_supervised and not self.patchnce
                and _positive(image, "source_contrastive_weight", .1),
            "patchnce": self.decoder and self.patchnce and _positive(image.get("patchnce") or {}, "weight", .15),
        }
        self.guard = bool((config.get("gradient_guard") or {}).get("enabled", False))
        self.stage1a_probes = any(self.enabled[k] for k in
                                  ("latent_anchor", "null_preservation", "conditioned_preservation"))
        self.inactive_prefixes = []
        fields = {
            "latent_anchor": ("cat_anchor_loss", "dog_anchor_loss", "latent_anchor_weight"),
            "infoot_alignment": ("infoot_feature_loss", "alignment_weight", "weighted_alignment_", "alignment_to_"),
            "semantic_neighborhood": ("semantic_neighborhood", "weighted_neighborhood_"),
            "conditional_structure": ("conditional_structure", "weighted_conditional_structure_"),
            "projection_support": ("projection_support", "weighted_projection_support_"),
            "null_preservation": ("null_preservation", "weighted_null_preservation_"),
            "conditioned_preservation": ("conditioned_preservation", "weighted_conditioned_preservation_"),
            "code_consistency": ("code_consistency", "weighted_code_consistency_", "applied_code_consistency_"),
            **{f"matching_{part}": (f"matching_{part}_loss", f"weighted_matching_{part}_")
               for part in ("variance", "covariance")},
            **{f"{d}_reconstruction": (f"{d}_reconstruction_loss",) for d in ("cat", "dog")},
        }
        for objective, prefixes in fields.items():
            if not self.enabled[objective]:
                self.inactive_prefixes.extend(prefixes)
        if not any(self.enabled[f"matching_{p}"] for p in ("variance", "covariance")):
            self.inactive_prefixes.extend(("matching_regularization", "weighted_matching_regularization_"))
        if not any(self.enabled[f"matching_{p}_contrastive"] for p in ("neighborhood", "conditional")):
            self.inactive_prefixes.extend(("matching_contrastive", "weighted_matching_contrastive_"))
        if not self.decoder:
            self.inactive_prefixes.append("decoded_translation")

    def _scalars(self, values):
        for key in list(values):
            if key.startswith(tuple(self.inactive_prefixes)):
                del values[key]

    @staticmethod
    def _diagnostic(values, old, new):
        if old in values:
            values.setdefault("diagnostics", {})[new] = values.pop(old)

    def _decoded(self, decoded):
        if self.self_supervised:
            return  # No external diagnostics are retained for this experiment.
        blocks = [decoded] + [decoded[d] for d in ("cat_to_dog", "dog_to_cat") if d in decoded]
        for block in blocks:
            self._diagnostic(block, "perceptual_cosine_distance", "perceptual_cosine_distance")
            if not self.enabled["structure"]:
                self._diagnostic(block, "structure_loss", "structure_cosine_distance")
                block.pop("structure_weight", None)
                block.pop("structure_effective_weighted_loss", None)
            for objective in ("perceptual", "structure_contrastive", "code_consistency"):
                if not self.enabled[objective]:
                    for key in list(block):
                        if key.startswith(objective):
                            del block[key]
            if not self.enabled["color_histogram"]:
                # Weight-zero controls still measure color on the same validation images.
                self._diagnostic(block, "color_histogram_loss", "color_histogram_distance")
                self._diagnostic(block, "per_image_color_histogram_loss", "per_image_color_histogram_distance")
                block.pop("color_histogram_weight", None)
                block.pop("color_histogram_effective_weighted_loss", None)

    def _conflicts(self, report):
        e = self.enabled
        structure = e["structure"] or e["structure_contrastive"]
        components = {
            "conditional": e["conditional_structure"], "infoot": e["infoot_alignment"],
            "protection": e["matching_variance"] or e["matching_covariance"],
            "perceptual": e["perceptual"], "structure": structure,
            "dino": e["perceptual"] or structure, "adversarial": e["adversarial"],
            "color": e["color_histogram"], "code": e["code_consistency"],
            "decoded": self.decoder, "translation": self.decoder,
            "reconstruction": e["cat_reconstruction"] or e["dog_reconstruction"],
            "matching": any(e[k] for k in ("infoot_alignment", "semantic_neighborhood", "conditional_structure",
                "projection_support", "matching_variance", "matching_covariance",
                "matching_neighborhood_contrastive", "matching_conditional_contrastive")),
        }
        for pairs in report.get("groups", {}).values():
            for name in list(pairs):
                first, second = name.split("_vs_")
                if not components[first] or not components[second] or (
                        name == "dino_vs_adversarial" and e["perceptual"] and not structure):
                    del pairs[name]
        if not e["code_consistency"]:
            report.pop("code_routing", None)

    def _clean_self_supervised(self, values):
        """Drop inactive legacy placeholders, including nested/window fields.

        Keep retrieval, original-RGB checks, geometry, PCGrad and solver health:
        they remain useful even though they are not additional objectives.
        """
        prefixes = ("perceptual_", "structure_", "adversarial_", "discriminator_", "dino_", "teacher_",
                    "decoded_discriminator_", "color_histogram", "code_consistency",
                    "conditional_structure", "semantic_neighborhood", "projection_support",
                    "null_preservation", "conditioned_preservation", "matching_contrastive")
        unused = {"primary_objective", "auxiliary_objective", "infoot_restart", "code_routing",
                  "cat_fake_score", "cat_real_score", "dog_fake_score", "dog_real_score"}
        if self.patchnce:
            prefixes += ("source_contrastive",)
            unused.add("source_negative_bank_ids")
        else:
            prefixes += ("patchnce",)
        for key in list(values):
            objective_key = key.removeprefix("weighted_").removeprefix("applied_")
            if objective_key.startswith(prefixes) or key in unused:
                del values[key]
            elif isinstance(values[key], dict):
                self._clean_self_supervised(values[key])
                if not values[key]:
                    del values[key]

    def format(self, metrics):
        """Copy a completed train/validation record, preserving measured numbers."""
        result = deepcopy(metrics)
        self._scalars(result)
        if "window_mean" in result:
            self._scalars(result["window_mean"])
        if not self.guard:
            result.pop("gradient_guard", None)
        if not self.stage1a_probes:
            for key in list(result):
                if "stage1a" in key and key.endswith("reconstruction"):
                    del result[key]
        if "decoded_translation" in result:
            self._decoded(result["decoded_translation"])
        if "gradient_conflicts" in result:
            self._conflicts(result["gradient_conflicts"])
        protection = result.get("matching_regularization", {})
        if protection and not self.enabled["matching_covariance"]:
            for block in [protection] + [protection[d] for d in ("cat", "dog") if d in protection]:
                self._diagnostic(block, "covariance_loss", "mean_squared_offdiagonal_covariance")
                block.pop("weighted_covariance_loss", None)
                block.pop("covariance_weight", None)
        contrast = result.get("matching_contrastive", {})
        for part in ("neighborhood", "conditional"):
            if contrast and not self.enabled[f"matching_{part}_contrastive"]:
                for key in list(contrast):
                    if key.startswith((part, f"weighted_{part}")):
                        del contrast[key]
        if self.self_supervised:
            self._clean_self_supervised(result)
        result["log_schema_version"] = 3 if self.self_supervised else 2
        result["enabled_losses"] = [name for name, enabled in self.enabled.items() if enabled]
        return result
