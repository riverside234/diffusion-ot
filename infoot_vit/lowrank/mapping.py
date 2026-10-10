"""Saved-factor grouped projection; no OT/kernel fitting during inference."""
import json
import math
from pathlib import Path
import time
import torch

from ..infoot_helper.feature_bank import FeatureBank, compatible_banks, digest, checked_file, validate_maps, file_hash, write_json
from ..infoot_helper.sampling import sample_ids, sampling_record
from ..infoot_helper.conditional import BalancedModel, normalize_rows
from ..infoot_helper.mapping import FeatureMapper, MappingResult, select_images
from .config import SCHEMA, MODE,PARTIAL_SCHEMA,PARTIAL_MODE
from ..infoot_helper.device import resolve_device,move
from .projection import project_scores
from .storage import validate_factors
from .experiment import validate_kernels
from .kernels import features


class LowRankMapper(FeatureMapper):
    """Reuse existing cache/report/PDAE interfaces; replace only patch algebra."""
    def _save_mapping(self,bank,output,ids,result):
        manifest = super()._save_mapping(bank,output,ids,result)
        manifest["lowrank_projection_dependency_sha256"] = {
            p.name:file_hash(p) for p in Path(__file__).parent.glob("*.py")}
        manifest["artifact_id"] = digest({k:v for k,v in manifest.items() if k != "artifact_id"})
        write_json(output/"manifest.json",manifest)
        return manifest

    @classmethod
    def load(cls,directory,*,device=None):
        directory = Path(directory).resolve()
        manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("schema") not in {SCHEMA,PARTIAL_SCHEMA} or manifest.get("status") != "complete"
                or digest({k:v for k,v in manifest.items() if k != "artifact_id"}) != manifest.get("artifact_id")):
            raise ValueError("Low-rank artifact is incomplete or its fingerprint changed.")
        partial = manifest["schema"] == PARTIAL_SCHEMA
        expected_files = {"image","kernels","samples","pair_selection","pairs"} if partial else {"image","kernels","samples","factors"}
        if set(manifest["files"]) != expected_files or manifest["config"]["mode"] != (PARTIAL_MODE if partial else MODE):
            raise ValueError("Incomplete low-rank artifact inventory.")
        resolve_device(device if device is not None else manifest["config"].get("device","cpu"))
        banks = []
        sampling = manifest["config"]["sampling"]
        for name in ("source","target"):
            bank = FeatureBank.load(directory/manifest[f"{name}_bank"]["path"])
            ids = sample_ids(bank.ids,sampling["images_per_domain"],sampling["seed"])
            if (bank.artifact_id != manifest[f"{name}_bank"]["artifact_id"] or ids != manifest[f"{name}_ids"]
                    or sampling_record(bank.ids,ids,sampling["seed"]) != manifest["sampling"][name]):
                raise ValueError("Low-rank bank identity or sampled support changed.")
            banks.append(bank.subset(ids))
        compatible_banks(*banks)
        if partial:
            from .partial_mapping import LowRankPartialMapper
            return LowRankPartialMapper(directory,manifest,*banks,device=device)
        return cls(directory,manifest,*banks,device=device)

    def _initialize(self,directory,manifest,source,target,device):
        self.directory,self.manifest,self.source,self.target = directory,manifest,source,target
        self.config,self.mode = manifest["config"],manifest["config"]["mode"]
        self.device = resolve_device(device if device is not None else self.config.get("device","cpu"))
        self.x,self.y = source.features.to(device=self.device,dtype=torch.float64),target.features.to(device=self.device,dtype=torch.float64)
        self.is_partial = self.mode == PARTIAL_MODE
        self.source_index = {sid:i for i,sid in enumerate(source.ids)}
        self.target_index = {tid:i for i,tid in enumerate(target.ids)}
        files = {key:checked_file(directory,entry) for key,entry in manifest["files"].items()}
        image = torch.load(files["image"],weights_only=True)
        if image["status"] != "converged": raise ValueError("Nonconverged image router.")
        self.image = BalancedModel(self.x.reshape(len(self.x),-1),self.y.reshape(len(self.y),-1),image)
        self.patch = None
        return files

    def __init__(self,directory,manifest,source,target,*,device=None):
        files = self._initialize(directory,manifest,source,target,device)
        n,m = len(self.x)*self.x.shape[1],len(self.y)*self.y.shape[1]
        state = torch.load(files["factors"],weights_only=True)
        self.factor_validation = validate_factors(state,self.config["optimizer"],(n,m,self.config["transport_rank"]))
        if state["fit_fingerprint"] != manifest["fit_fingerprint"] or state["solver_report"]["status"] != "converged_sampled_objective":
            raise ValueError("Invalid factor fit identity or convergence status.")
        q,r,g = (state[key].to(device=self.device,dtype=torch.float64) for key in ("q","r","g"))
        kernel = torch.load(files["kernels"],weights_only=True)
        validate_kernels(kernel,(n,m,self.config["kernel_rank"]),self.x.shape[-1])
        self.kernel_source = move(kernel["source"],self.device)
        fx,fy = kernel["fx"].to(self.x),kernel["fy"].to(self.y)
        self.cross = ((fx.T@q)/g) @ (fy.T@r).T  # Only [kernel_rank,kernel_rank].
        self.density_y = (fy@fy.mean(0)).reshape(len(self.y),self.y.shape[1])
        self.target_kernel_features = fy.reshape(len(self.y),self.y.shape[1],-1)
        if (self.density_y <= 0).any() or not torch.isfinite(self.cross).all():
            raise ValueError("Invalid saved-factor KDE density.")

    @torch.no_grad()
    def map_features(self,values,query_ids,*,valid_mask=None,return_metadata=False,chunk_size=None,on_query=None):
        validate_maps(values,query_ids,self.source.representation,valid_mask)
        size = self.config["projection"]["query_chunk_size"] if chunk_size is None else chunk_size
        if type(size) is not int or size < 1: raise ValueError("chunk_size must be positive.")
        start = time.perf_counter(); outputs,records = [],[]
        settings = self.config["projection"]
        for offset in range(0,len(values),size):
            for patches,qid in zip(values[offset:offset+size].to(device=self.device,dtype=torch.float64),query_ids[offset:offset+size]):
                raw = self.image.conditional_weights(patches.reshape(1,-1))[0]
                alpha,retained = select_images(raw,self.target.ids,qid,settings)
                left = features(patches,self.kernel_source)@self.cross
                mapped = torch.zeros_like(patches)
                effective,residual = 0.,0.
                for j in range(0,len(self.y),settings["target_chunk_size"]):
                    stop = j+settings["target_chunk_size"]
                    active = alpha[j:stop] > 0
                    if not active.any(): continue
                    # Sparse routing excludes these groups entirely, including
                    # conditional validity checks and projection diagnostics.
                    weights = alpha[j:stop][active]
                    candidate,diag = project_scores(left,self.target_kernel_features[j:stop][active],self.y[j:stop][active],self.density_y[j:stop][active],settings.get("patch_chunk_size",64))
                    if (diag["weight_row_sums"] == 0).any():
                        raise ValueError("Zero-mass balanced conditional row; inspect kernel approximation.")
                    mapped += (weights[:,None,None]*candidate).sum(0)
                    residual = max(residual,float((diag["weight_row_sums"]-1).abs().max()))
                    effective += float((weights*diag["patch_entropy"].exp().mean(-1)).sum())
                if not torch.isfinite(mapped).all(): raise ValueError("Nonfinite low-rank mapped features.")
                entropy = float(-(alpha*alpha.clamp_min(1e-300).log()).sum())
                record = dict(query_id=qid,image_weights=alpha.tolist(),target_ids=self.target.ids,
                    image_entropy=entropy,image_effective_targets=math.exp(entropy),image_max_weight=float(alpha.max()),
                    image_normalized_entropy=entropy/math.log(len(alpha)) if len(alpha)>1 else 0.,
                    top_k_retained_mass=retained,top_k_discarded_mass=1-retained,
                    within_image_effective_patches=effective,group_mass_residual=residual,
                    transport_rank=self.config["transport_rank"],kernel_rank=self.config["kernel_rank"])
                outputs.append(mapped); records.append(record)
                if on_query: on_query(record)
        result = torch.stack(outputs).to(values)
        metadata = MappingResult(result,torch.ones(result.shape[:2],dtype=values.dtype,device=values.device),
            torch.ones(result.shape[:2],dtype=torch.bool,device=values.device),
            dict(mode=self.mode,queries=records,seconds=time.perf_counter()-start,device=str(self.device)))
        return metadata if return_metadata else result
