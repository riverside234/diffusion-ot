"""Pair factor projection using the existing partial router/mask accounting."""
from collections import OrderedDict
import json
import torch

from .mapping import LowRankMapper
from .config import optimizer_config
from .storage import validate_factors
from . import pair_kernels,kernels
from .projection import partial_project
from ..infoot_helper.mapping import FeatureMapper
from ..infoot_helper.feature_bank import checked_file
from ..infoot_helper.pair_selection import selection_edges
from ..infoot_helper.device import move


class LowRankPartialMapper(LowRankMapper):
    map_features = FeatureMapper.map_features

    def __init__(self,directory,manifest,source,target,*,device=None):
        files = self._initialize(directory,manifest,source,target,device)
        self.options = optimizer_config(self.config)
        selection = json.loads(files["pair_selection"].read_text(encoding="utf-8"))
        if selection["router_sha256"] != manifest["files"]["image"]["sha256"]:
            raise ValueError("Partial selection router identity mismatch.")
        expected = selection_edges(selection,source.ids,target.ids,self.config["fit_pair_top_k"])
        self.pairs = {}
        for line in files["pairs"].read_text(encoding="utf-8").splitlines():
            entry = json.loads(line); key = entry["source_id"],entry["target_id"]
            if (key in self.pairs or entry["source_index"] != self.source_index.get(key[0])
                    or entry["target_index"] != self.target_index.get(key[1])):
                raise ValueError("Invalid partial factor inventory IDs/order.")
            checked_file(directory,entry)
            self.pairs[key] = entry
        if set(self.pairs) != expected or len(expected) != manifest["pair_count"]:
            raise ValueError("Selected partial factors are missing or unexpected; inference never fits replacement pairs.")
        self.neighbors = {sid:[tid for tid in target.ids if (sid,tid) in expected] for sid in source.ids}
        self.pair_mask = torch.zeros(len(source.ids),len(target.ids),device=self.device,dtype=torch.bool)
        for sid,tid in expected:
            self.pair_mask[self.source_index[sid],self.target_index[tid]] = True
        kernel = torch.load(files["kernels"],map_location="cpu",weights_only=True)
        for name,images in (("source",self.x),("target",self.y)):
            pair_kernels.validate_collection(kernel[name],images.shape,self.config["kernel_rank"])
        self.kernel = move(kernel,self.device)
        self.fx,self.fy = self.kernel["source"]["factors"].double(),self.kernel["target"]["factors"].double()
        self._factor_cache = OrderedDict()  # Bounded GPU cache, never all pair factors.

    def _partial_query(self,patches,i):
        return kernels.features(patches,pair_kernels.parameters(self.kernel["source"],i))

    def _partial_projection(self,patches,i,j,sid,tid,context):
        key = sid,tid
        if key not in self._factor_cache:
            state = torch.load(checked_file(self.directory,self.pairs[key]),map_location="cpu",weights_only=True)
            constraints = validate_factors(state,self.options,(self.x.shape[1],self.y.shape[1],self.config["transport_rank"]))
            if (state["fit_fingerprint"] != self.manifest["fit_fingerprint"] or state["source_id"] != sid
                    or state["target_id"] != tid or state["source_index"] != i or state["target_index"] != j
                    or state["solver_report"]["status"] != "converged_sampled_objective"
                    or state["objective_version"] != "transported_kde_mi_mass_weighted_v1"):
                raise ValueError("Partial factor identity/objective/status mismatch.")
            self._factor_cache[key] = (tuple(state[k].to(self.x) for k in ("q","r","g")),constraints["storage_relative_tolerance"])
            if len(self._factor_cache) > 32:
                self._factor_cache.popitem(last=False)
        self._factor_cache.move_to_end(key)
        factors,tolerance = self._factor_cache[key]
        candidate,g,diag = partial_project(context,self.fx[i],self.fy[j],self.y[j],factors,
            self.kernel["source"]["support"][i],self.config["projection"]["patch_chunk_size"])
        return candidate,g,diag,tolerance
