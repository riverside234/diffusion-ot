import torch

from diffusion_ot.models.matching_head import matching_geometry_diagnostics
from ..batchnorm_matching import add_matching_features
from ..encoding import encode_batches
from ..infoot_cotraining_helper import alignment_loss, conditional_mapping, fit_transport
from ..native_flow import native_flow_loss
from ..translation_contrastive import translation_contrastive_loss


class CoTrainingStep(torch.nn.Module):
    def __init__(self, domains, batch_norms, settings):
        super().__init__()
        self.branches = torch.nn.ModuleDict({name: context.branch for name, context in domains.items()})
        self.batch_norms = batch_norms
        self.domains, self.settings = domains, settings

    def forward(self, latents, report=False):
        settings = self.settings
        encoded = encode_batches(
            self.domains, latents, query_count=settings["query_count"],
            encode_batch_size=settings["encode_batch_size"],
        )
        add_matching_features(encoded, self.batch_norms)
        flow = {
            name: native_flow_loss(context, encoded[name]["x0"][:settings["flow_batch_size"]],
                                   encoded[name]["m"][:settings["flow_batch_size"]])
            for name, context in self.domains.items()
        }
        references = {name: batch["references"]["v"] for name, batch in encoded.items()}
        matching = {name: batch["references"]["m"] for name, batch in encoded.items()}
        plan = fit_transport(
            matching["cat"], matching["dog"], h=settings["fit_h"],
            mi_weight=settings["mi_weight"], reg=settings["reg"], iterations=settings["fit_iterations"],
        )
        loss_infoot = alignment_loss(
            matching["cat"], matching["dog"], plan, h=settings["fit_h"],
            mi_weight=settings["mi_weight"], reg=settings["reg"],
        )
        contrastive = []
        for source, target, coupling in (("cat", "dog", plan), ("dog", "cat", plan.T)):
            mapped_m = conditional_mapping(
                encoded[source]["queries"]["m"], matching[source], matching[target], coupling,
                h=settings["projection_h"],
            )
            loss, _ = translation_contrastive_loss(
                self.domains[target], encoded[source], mapped_m,
                batch_norm=self.batch_norms[target], reference_v=references[target],
                steps=settings["sampling_steps"],
            )
            contrastive.append(loss)
        loss_contrastive = torch.stack(contrastive).mean()
        loss = (settings["flow_weight"] * torch.stack(list(flow.values())).mean()
                + settings["infoOT_loss_weight"] * loss_infoot
                + settings["contrastive_weight"] * loss_contrastive)
        metrics = {}
        if report:
            metrics = {"flow_cat": flow["cat"], "flow_dog": flow["dog"], "infoot": loss_infoot,
                       "contrastive": loss_contrastive, "total": loss}
            metrics = {key: value.detach().item() for key, value in metrics.items()}
            for name, v in references.items():
                geometry = matching_geometry_diagnostics(v, v)
                metrics.update({f"rank_{name}": geometry["matching_covariance_effective_rank"],
                                f"max_rank_{name}": min(len(v) - 1, v.shape[1]),
                                f"var_{name}": geometry["raw_code_variance"]})
        return loss, metrics
