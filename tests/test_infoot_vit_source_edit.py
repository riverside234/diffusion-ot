"""Source-latent math, identity guards and a full synthetic cached-v9 run."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from test_infoot_vit_averaging import completed_v8, completed_v7, experiment, banks, cpu_threads, fake_umap
from test_infoot_vit_projection_overrides import hashes, forbid_fitting
from infoot_vit import compare_displacement as old_runner, edit_source
from infoot_vit.diagnose_blur import tensor_hash
from infoot_vit.infoot_helper.feature_bank import file_hash, save_tensor, write_json
from infoot_vit.infoot_helper.source_edit import edit_schedule, encode_source, flow_edit, make_velocity, step_noises
from diffusion_ot.evaluation.stage1a_eval import integrate_pdae_flow as real_integrate


def inputs():
    g = torch.Generator().manual_seed(61)
    x = torch.randn(2, 4, 3, 3, generator=g, dtype=torch.float64)
    src = torch.ones(2, 4, 3, dtype=x.dtype)
    dst = src * 3
    valid = torch.tensor([[True, True, False, True], [True, False, True, False]])
    return x, src, dst, valid, ["a", "b"]


def test_constant_velocity_analytic_solution_and_positive_time_direction():
    x, src, dst, mask, ids = inputs()
    before = x.clone()
    calls = []
    def velocity(value, time, tokens, valid):
        calls.append((value.clone(), time, tokens.clone(), valid.clone()))
        return torch.ones_like(value) * tokens[:, 0, 0, None, None, None]
    actual, trace = flow_edit(x, src, dst, mask, ids, seed=42, num_steps=40, velocity=velocity)
    torch.testing.assert_close(actual, x + 1, atol=3e-15, rtol=1e-14)
    assert torch.equal(x, before) and len(calls) == 40 and len(trace) == 20
    assert trace[0]["time"] == .5125 and trace[-1]["time"] == .9875
    assert all(torch.equal(call[3], mask) for call in calls)
    torch.testing.assert_close(calls[0][0], calls[1][0], atol=0, rtol=0)
    assert all(row["delta_velocity_rms"] == 2 for row in trace)


def test_equal_conditions_are_exact_identity_even_for_nonlinear_velocity():
    x, src, _, mask, ids = inputs()
    result, trace = flow_edit(x, src, src, mask, ids, seed=12, num_steps=8,
        velocity=lambda x,t,z,m: x.sin() * t + z[:,0,0,None,None,None])
    assert torch.equal(result, x) and all(row["delta_velocity_rms"] == 0 for row in trace)


def test_noise_independence_reordering_and_chunked_edit_agree():
    x, src, dst, mask, ids = inputs()
    noise = step_noises(x, ids, 7, 20)
    assert not torch.equal(noise, step_noises(x, ids, 7, 21))
    assert torch.equal(noise.flip(0), step_noises(x.flip(0), ids[::-1], 7, 20))
    def vel(x,t,z,m):return .1*x.square() + .2*z[:,0,0,None,None,None]
    full,_ = flow_edit(x, src, dst, mask, ids, seed=7, num_steps=8, velocity=vel)
    chunks = [flow_edit(x[i:i+1],src[i:i+1],dst[i:i+1],mask[i:i+1],ids[i:i+1],
        seed=7,num_steps=8,velocity=vel)[0] for i in range(2)]
    assert torch.equal(full, torch.cat(chunks))


@pytest.mark.parametrize("steps,start", [(0,.5),(3,.5),(40,1.),(40,float('nan')),(40,-.1)])
def test_bad_schedules_rejected(steps,start):
    with pytest.raises(ValueError):edit_schedule(steps,start)


@pytest.mark.parametrize("bad", ["empty_mask","nan_velocity","shape_velocity","nan_latent"])
def test_invalid_states_fail_without_a_fallback(bad):
    x,src,dst,mask,ids=inputs()
    def vel(x,t,z,m):return torch.zeros_like(x)
    if bad == "empty_mask":mask[0]=False
    if bad == "nan_latent":x[0,0,0,0]=float('nan')
    if bad == "nan_velocity":vel=lambda *a: torch.full_like(x,float('nan'))
    if bad == "shape_velocity":vel=lambda *a: x[:1]
    with pytest.raises(ValueError):flow_edit(x,src,dst,mask,ids,seed=7,num_steps=8,velocity=vel)


class TinyVAE(torch.nn.Module):
    config = SimpleNamespace(scaling_factor=.25, shift_factor=None)
    def encode(self, x):
        value = torch.cat([F.interpolate(x, size=(32,32)), torch.zeros(len(x),1,32,32,device=x.device)],1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:value,
            sample=lambda:pytest.fail("Source VAE encoding must not sample its posterior")))


def test_vae_uses_posterior_mode_range_and_scale():
    images=torch.full((2,3,8,8),.75)
    latent=encode_source(TinyVAE(),images,device="cpu",dtype=torch.float64,batch_size=1)
    assert latent.shape==(2,4,32,32) and latent.dtype==torch.float64
    assert torch.equal(latent[:,:3],torch.full_like(latent[:,:3],.125))
    assert not latent[:,3].any()
    with pytest.raises(ValueError):encode_source(TinyVAE(),images*2,device="cpu",dtype=torch.float32,batch_size=1)


class TinyBranch(torch.nn.Module):
    semantic_cfg_enabled=True
    def predict_cfg_with_z(self, *, x_t, timestep, z, guidance_scale, class_labels, condition_padding_mask):
        assert (class_labels==1000).all()
        kept=z.masked_fill(condition_padding_mask[...,None],0).sum((1,2))
        return SimpleNamespace(sample=.01*x_t + .001*guidance_scale*kept[:,None,None,None]
                               + .01*timestep[:,None,None,None])


def test_cfg_velocity_matches_existing_sampler_including_masks_and_chunking():
    x,src,_,mask,_=inputs(); e=SimpleNamespace(branch=TinyBranch(),transformer=object())
    velocity=make_velocity(e,guidance_scale=1.5,null_label=1000,batch_size=1)
    derivative=velocity(x,.5125,src,mask)
    result=real_integrate(e.branch,e.transformer,x,src,num_steps=1,start_time=.5,end_time=.525,
        guidance_scale=1.5,null_label=1000,batch_size=1,condition_padding_mask=~mask)
    torch.testing.assert_close(result,x+.025*derivative,atol=1e-14,rtol=1e-14)


def test_attention_is_invariant_to_token_order_but_not_unequal_multiplicity():
    from diffusion_ot.models.pdae_v2.cross_attention import ImageCrossAttention
    torch.manual_seed(41)
    attention=ImageCrossAttention(8,token_dim=3,num_heads=2).double()
    torch.nn.init.normal_(attention.out.weight)
    hidden=torch.randn(2,5,8,dtype=torch.float64); tokens=torch.randn(2,4,3,dtype=torch.float64)
    mask=torch.tensor([[False,False,True,False],[False,True,False,False]])
    permutation=torch.tensor([2,0,3,1]); expected=attention(hidden,tokens,mask)
    assert expected.abs().max()>0
    torch.testing.assert_close(attention(hidden,tokens[:,permutation],mask[:,permutation]),expected,atol=1e-13,rtol=1e-13)
    assert not torch.allclose(attention(hidden,torch.cat([tokens,tokens[:,:1]],1),
        torch.cat([mask,mask[:,:1]],1)),expected)


@pytest.fixture
def completed_v9(completed_v8,monkeypatch,fake_umap):
    args,_,loaded,sampled=completed_v8
    reference=old_runner.run_experiment(args)
    args.reference_run=args.output_dir
    args.output_dir=args.output_dir.parent/'source_edit'
    args.work_dir=args.work_dir.parent/'source_edit_work'
    args.reference_work_dir=None
    from diffusion_ot.evaluation import stage1a_eval
    evaluator=stage1a_eval.load_stage1a_evaluator()
    class Encoder(torch.nn.Module):
        snapshot_identity=evaluator.branch.encoder.snapshot_identity
        architecture_spec=evaluator.branch.encoder.architecture_spec
        def forward(self,images):return F.adaptive_avg_pool2d(images,(2,2)).flatten(2).transpose(1,2)
    evaluator.branch.encoder=Encoder()
    evaluator.branch.semantic_cfg_enabled=True
    evaluator.branch.predict_cfg_with_z=TinyBranch().predict_cfg_with_z
    evaluator.vae=TinyVAE()
    calls=[]
    def integrate(*a,**k):
        # The real sampler recursively invokes its module entry point for chunks.
        if len(a[2]) == len(reference['generation']['query_ids']):
            calls.append((a[2].clone(),a[3].clone(),k))
        return real_integrate(*a,**k)
    monkeypatch.setattr(stage1a_eval,'integrate_pdae_flow',integrate)
    monkeypatch.setattr(stage1a_eval,'decode_vae_latents',lambda vae,z,**kw:
        ((F.interpolate(z[:,:3]/.25,size=(8,8))+1)/2).clamp(0,1))
    loaded.clear();sampled.clear();fake_umap.clear()
    return args,reference,loaded,calls


def test_full_source_edit_reuses_cache_no_fit_or_mapping_and_three_files(completed_v9,monkeypatch,fake_umap):
    args,reference,loaded,calls=completed_v9
    from infoot_vit.infoot_helper.mapping import FeatureMapper
    forbid_fitting(monkeypatch)
    monkeypatch.setattr(FeatureMapper,'project_bank',lambda *a,**k:pytest.fail("Do not remap"))
    before=hashes(args.mapping),hashes(args.reference_run)
    result=edit_source.run_experiment(args)
    assert result['status']=='complete' and not result['quality_review']['automatic_quality_pass']
    assert result['initial_noise_sha256']==reference['initial_noise_sha256']
    assert result['common_valid_mask_sha256']==reference['common_valid_mask_sha256']
    assert len(loaded)==1 and len(calls)==4 and len(result['row_order'])==8
    assert torch.equal(calls[0][0],calls[1][0]) and torch.equal(calls[2][0],calls[3][0])
    assert torch.equal(calls[1][1],calls[3][1])
    for i in range(4):assert torch.equal(calls[0][2]['condition_padding_mask'],calls[i][2]['condition_padding_mask'])
    assert all(calls[i][2]['num_steps']==20 and calls[i][2]['start_time']==.5 for i in (2,3))
    cache=torch.load(Path(result['output']['file']),weights_only=True)
    torch.testing.assert_close(cache['noised_source_initial_state'],.5*(cache['initial_noise']+cache['source_latents']))
    assert len(result['flow_edit_trace'])==20 and len(fake_umap)==2
    assert result['umap']['schema']=='infoot_generated_image_comparison_umap_v1'
    assert result['umap']['style']['mapped_label']=='Generated images'
    assert all(p['valid_tokens']==4 for p in result['umap']['panels']['image']['groups']['source_query']['points'])
    assert set(result['generated_readouts'])==set(edit_source.METHOD_TITLES)
    features=torch.load(result['generated_feature_cache']['file'],weights_only=True)
    for name,identity in result['generated_readouts'].items():
        expected=F.adaptive_avg_pool2d(cache['images'][name]*2-1,(2,2)).flatten(2).transpose(1,2)
        assert torch.equal(features['features'][name],expected)
        assert identity['features_sha256']==tensor_hash(expected)
    assert {p.name for p in args.output_dir.iterdir()}=={'experiment_report.json','comparison_grid.png','umap.png'}
    assert (hashes(args.mapping),hashes(args.reference_run))==before
    assert result['output']['sha256']==file_hash(result['output']['file'])


@pytest.mark.parametrize('change',['checkpoint','cache','query','reference','mask','noise','config','missing_cache'])
def test_changed_inputs_fail_before_loading_decoder(completed_v9,change):
    args,ref,loaded,calls=completed_v9
    cache=Path(ref['output']['file'])
    if change=='checkpoint':args.checkpoint.write_bytes(b'wrong')
    elif change=='cache':cache.write_bytes(b'wrong')
    elif change=='query':args.query_bank=args.query_bank.parent/'source'
    elif change=='reference':ref['status']='failed'
    elif change=='mask':ref['common_valid_mask_sha256']='0'*64
    elif change=='noise':ref['generation']['per_image_noise_seeds'][0]+=1
    elif change=='config':args.eval_config.write_text('changed: true\n')
    elif change=='missing_cache':args.reference_work_dir=args.output_dir.parent/'missing'
    write_json(args.reference_run/'experiment_report.json',ref)
    with pytest.raises((ValueError,FileNotFoundError)):edit_source.run_experiment(args)
    assert not loaded and not calls and not args.output_dir.exists()


def test_failed_output_umap_preserves_generated_cache_and_failed_report(completed_v9,monkeypatch):
    args,_,_,_=completed_v9
    monkeypatch.setattr(edit_source.comparison_umap,'save_comparison_umap',
        lambda *a,**k: (_ for _ in ()).throw(RuntimeError('plot failed')))
    with pytest.raises(RuntimeError,match='plot failed'):edit_source.run_experiment(args)
    report=json.loads((args.output_dir/'experiment_report.json').read_text())
    assert report['status']=='failed' and Path(report['output']['file']).is_file()
    assert Path(report['generated_feature_cache']['file']).is_file()


def test_generated_image_umap_rejects_conditioning_masks(banks,tmp_path,fake_umap):
    from test_infoot_vit_umap import example
    mapper,query,result=example(banks)
    with pytest.raises(ValueError,match='full re-encoded'):
        edit_source.comparison_umap.save_comparison_umap(mapper,query,
            {name:result for name in edit_source.METHOD_TITLES},tmp_path,
            mapped_ids={name:name for name in edit_source.METHOD_TITLES},
            method_titles=edit_source.METHOD_TITLES,readout='generated_images')
    assert not fake_umap
