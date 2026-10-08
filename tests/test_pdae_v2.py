"""Offline checks of v2 conditioning, pretrained equivalence, and the Stage 1A loop."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from test_pdae_sit_shapes import FakeSiT, FakeAttention
from test_stage1a_refinement import setup as latent_training_setup
from test_decoded_translation import TinyVAE
from diffusion_ot.models.pdae_sit import build_pdae_sit_branch
from diffusion_ot.models.pdae_v2.branch import PDAEV2Branch, TokenConditionedSiT
from diffusion_ot.models.pdae_v2.cross_attention import ImageCrossAttention
from diffusion_ot.models.pdae_v2.encoder import (
    FrozenSiglipPatchEncoder, MANIFEST_NAME, MODEL_ID, file_sha256, snapshot_identity,
)
from diffusion_ot.training.train_pdae_domain import TrainableEMA, _optimizer_groups


@pytest.fixture(autouse=True)
def limited_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


class NativeSiTBlock(nn.Module):
    """The existing SiT's six-way AdaLN block, with nonzero gates/weights."""
    def __init__(self, hidden_size=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = FakeAttention(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, 2 * hidden_size), nn.GELU(approximate="tanh"),
                                 nn.Linear(2 * hidden_size, hidden_size))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x, c):
        a, b, g, d, e, h = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + g.unsqueeze(1) * self.attn(self.norm1(x) * (1 + b.unsqueeze(1)) + a.unsqueeze(1))
        return x + h.unsqueeze(1) * self.mlp(self.norm2(x) * (1 + e.unsqueeze(1)) + d.unsqueeze(1))


def base_model():
    model = FakeSiT()
    model.blocks = nn.ModuleList([NativeSiTBlock(), NativeSiTBlock()])
    nn.init.normal_(model.pos_embed, std=.1)
    return model


class TinyFrozenEncoder(nn.Module):
    """Small RGB stand-in for expensive pretrained tokens in loop tests only."""
    architecture_spec = {"kind": "siglip2_vit_b16", "token_dim": 768, "frozen": True}
    snapshot_identity = {"model_id": MODEL_ID, "revision": "offline-test-snapshot"}

    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 768, 4, 4).requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, rgb):
        return self.patch(rgb).flatten(2).transpose(1, 2)


def branch(base=None):
    return PDAEV2Branch(TinyFrozenEncoder(), TokenConditionedSiT(
        base or base_model(), num_heads=2, lora_rank=2, lora_alpha=2), image_size=8)


def inputs():
    return dict(x0_latent=torch.randn(2, 4, 8, 8), x_t=torch.randn(2, 4, 8, 8),
                timestep=torch.tensor([.05, .8]), class_labels=torch.tensor([1, 10]),
                encoder_image=torch.rand(2, 3, 8, 8) * 2 - 1)


def activate_cross_attention(model):
    for attention in model.semantic_transformer.image_attention:
        nn.init.normal_(attention.out.weight, std=.1)
        nn.init.constant_(attention.out.bias, .3)


@pytest.mark.parametrize("hidden_size,num_heads", [(768, 12), (1024, 16)])
def test_full_width_attention_accepts_sit_grid_and_siglip_patch_lengths(hidden_size, num_heads):
    attention = ImageCrossAttention(hidden_size, token_dim=768, num_heads=num_heads)
    nn.init.normal_(attention.out.weight, std=.02)
    hidden = torch.randn(1, 256, hidden_size, requires_grad=True)
    tokens = torch.randn(1, 196, 768, requires_grad=True)
    output = attention(hidden, tokens)
    assert output.shape == (1, 256, hidden_size) and torch.isfinite(output).all()
    output.square().mean().backward()
    assert torch.isfinite(hidden.grad).all() and hidden.grad.abs().sum() > 0
    assert torch.isfinite(tokens.grad).all() and tokens.grad.abs().sum() > 0


def test_pretrained_equivalence_and_zero_only_new_output_projections():
    torch.manual_seed(32)
    original = base_model().eval()
    model = branch(deepcopy(original)).eval()
    args = inputs()
    expected = original(args['x_t'], args['timestep'], args['class_labels']).sample
    output = model(**args)
    assert expected.abs().sum() > 0 and output.z.shape == (2, 4, 768)
    # Frozen/LoRA GEMMs can differ in last-bit rounding from the original GEMM.
    torch.testing.assert_close(output.sample, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(output.delta_sample, torch.zeros_like(expected), atol=1e-6, rtol=0)
    assert not hasattr(model.semantic_transformer, 'z_proj')
    assert not hasattr(model.semantic_transformer, 'final_adapter')
    for index, attention in enumerate(model.semantic_transformer.image_attention):
        assert torch.count_nonzero(attention.out.weight) == torch.count_nonzero(attention.out.bias) == 0
        assert attention.q.weight.abs().sum() > 0 and attention.k.weight.abs().sum() > 0
        assert attention.norm.weight.abs().sum() > 0
        if index:
            assert attention.q.weight is not model.semantic_transformer.image_attention[0].q.weight
    assert model.semantic_conditioner.null_token.abs().sum() > 0


def test_two_step_gradient_flow_train_modes_and_optimizer_groups():
    torch.manual_seed(6)
    model = branch().train()
    groups = _optimizer_groups(model, dict(lr_adapter=.001, lr_lora=.001))
    assert [g['group_name'] for g in groups] == ['adapter', 'lora']
    assert not model.encoder.training and not model.semantic_transformer.base.training
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW(groups)
    args = inputs()
    args['semantic_drop_mask'] = torch.tensor([False, True])
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(**args)
        assert torch.isfinite(output.sample).all()
        loss = (output.sample - torch.randn_like(output.sample)).square().mean()
        loss.backward()
        trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())
        attention = model.semantic_transformer.image_attention[0]
        assert attention.out.weight.grad.abs().sum() > 0
        if step == 0:
            assert attention.q.weight.grad.abs().sum() == 0
            assert model.semantic_transformer.feature_projector[1].weight.grad.abs().sum() == 0
        else:
            assert attention.q.weight.grad.abs().sum() > 0
            assert attention.k.weight.grad.abs().sum() > 0
            assert attention.norm.weight.grad.abs().sum() > 0
            assert model.semantic_transformer.feature_projector[1].weight.grad.abs().sum() > 0
            assert model.semantic_conditioner.null_token.grad.abs().sum() > 0
            assert any(p.grad.abs().sum() > 0 for n, p in trainable.items() if '.lora_down.' in n)
        optimizer.step()
    for name, p in model.named_parameters():
        if name in frozen:
            assert p.grad is None
            torch.testing.assert_close(p, frozen[name], atol=0, rtol=0)


def test_masked_padding_permutation_empty_rows_and_cfg():
    model = branch().eval()
    activate_cross_attention(model)
    args = inputs()
    z = torch.randn(2, 7, 768)  # different source/query lengths; no grid inferred
    mask = torch.tensor([[False, False, True, True, True, True, True], [True] * 7])
    predict = lambda code, padding, scale=1.: model.predict_cfg_with_z(
        args['x_t'], args['timestep'], code, class_labels=args['class_labels'],
        guidance_scale=scale, condition_padding_mask=padding).sample
    expected = predict(z, mask)
    dirty = z.masked_fill(mask[..., None], float('nan')).requires_grad_()
    actual = predict(dirty, mask)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert torch.isfinite(dirty.grad).all() and dirty.grad[mask].abs().sum() == 0
    original = model.semantic_transformer.base(args['x_t'], args['timestep'], args['class_labels']).sample
    torch.testing.assert_close(actual[1], original[1])  # all-padding residual incl. trained bias is zero
    order = torch.randperm(7)
    torch.testing.assert_close(predict(z[:, order], mask[:, order]), expected, atol=1e-6, rtol=1e-5)
    unconditional = predict(z, mask, 0)
    torch.testing.assert_close(unconditional, predict(model.semantic_null_like(z), mask))
    torch.testing.assert_close(predict(z, mask, 2), unconditional + 2 * (expected - unconditional))
    for bad in (mask.float(), mask[:, :3]):
        with pytest.raises(ValueError, match='padding_mask'):
            predict(z, bad)


def test_shared_projector_runs_once_and_cross_attention_precedes_ffn():
    model = branch().eval()
    order = []
    hooks = [model.semantic_transformer.feature_projector.register_forward_hook(lambda *a: order.append('project'))]
    for index, block in enumerate(model.semantic_transformer.base.blocks):
        hooks.append(model.semantic_transformer.image_attention[index].register_forward_hook(
            lambda *a, i=index: order.append(f'cross{i}')))
        hooks.append(block.mlp.register_forward_hook(lambda *a, i=index: order.append(f'ffn{i}')))
    model(**inputs())
    for hook in hooks:
        hook.remove()
    assert order == ['project', 'ffn0', 'cross0', 'ffn0', 'ffn1', 'cross1', 'ffn1']


def test_raw_ema_and_optimizer_roundtrip_and_mismatch_guards(tmp_path):
    torch.manual_seed(9)
    original_base = base_model()
    model = branch(deepcopy(original_base)).eval()
    optim = torch.optim.AdamW(_optimizer_groups(model, {'lr': .001}))
    ema = TrainableEMA(model, decay=.9)
    args = inputs()
    model(**args).sample.square().mean().backward()
    optim.step()
    ema.update(model)
    path = tmp_path / 'checkpoint.pt'
    torch.save(dict(model=model.pdae_state_dict(), optimizer=optim.state_dict(), ema=ema.state_dict(), step=1), path)
    saved = torch.load(path, weights_only=False)
    assert 'encoder' not in saved['model'] and saved['model']['format_version'] == 6
    restored = branch(deepcopy(original_base)).eval()
    # This test's frozen substitute, like the real model, must come from identical weights.
    restored.encoder.load_state_dict(model.encoder.state_dict())
    restored.load_pdae_state_dict(saved['model'])
    restored_optim = torch.optim.AdamW(_optimizer_groups(restored, {'lr': .001}))
    restored_optim.load_state_dict(saved['optimizer'])
    assert all(s['step'] == 1 for s in restored_optim.state.values())
    torch.testing.assert_close(restored(**args).sample, model(**args).sample, atol=0, rtol=0)
    restored_ema = TrainableEMA(restored, decay=.9)
    restored_ema.load_state_dict(saved['ema'], restored)
    with ema.average_parameters(model), restored_ema.average_parameters(restored):
        torch.testing.assert_close(restored(**args).sample, model(**args).sample, atol=0, rtol=0)
    bad = deepcopy(saved['model'])
    bad['frozen_encoder']['revision'] = 'different'
    with pytest.raises(ValueError, match='frozen_encoder'):
        restored.load_pdae_state_dict(bad)
    bad = deepcopy(saved['model'])
    bad['generator']['semantic_transformer']['architecture']['num_heads'] = 1
    with pytest.raises(ValueError, match='architecture mismatch'):
        restored.load_pdae_state_dict(bad)
    with pytest.raises(ValueError, match='format_version'):
        restored.load_pdae_state_dict({'format_version': 5})


def test_real_transformers_processor_and_dense_patch_output(tmp_path):
    transformers = pytest.importorskip('transformers')
    # Real Transformers classes, randomly initialized, small depth; no remote weights.
    config = transformers.SiglipVisionConfig(hidden_size=768, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=12, image_size=224, patch_size=16)
    vision = transformers.SiglipVisionModel(config)
    processor_class = getattr(transformers, 'SiglipImageProcessorPil', None) or transformers.SiglipImageProcessor
    processor = processor_class(size={'height': 224, 'width': 224})
    encoder = FrozenSiglipPatchEncoder(vision, processor, {'revision': 'test'})
    encoder.train()
    assert not encoder.training and not vision.training
    images = torch.rand(1, 3, 256, 256) * 2 - 1
    tokens = encoder(images)
    assert tokens.shape == (1, 196, 768) and not tokens.requires_grad
    pixels = processor(images=list(((images + 1) / 2).numpy()), do_rescale=False,
                       return_tensors='pt', input_data_format='channels_first')['pixel_values']
    with torch.no_grad():
        torch.testing.assert_close(tokens, vision(pixel_values=pixels).last_hidden_state)
    with pytest.raises(ValueError, match=r'in \[-1,1\]'):
        encoder(images + 3)


@pytest.mark.parametrize('damage', [None, 'missing_vision', 'mismatched_vision', 'unexpected_vision'])
def test_local_loading_from_composite_siglip_checkpoint(tmp_path, monkeypatch, damage):
    transformers = pytest.importorskip('transformers')
    # Official downloads contain both towers. Exercise actual key-prefix loading.
    config = transformers.SiglipConfig(vision_config=dict(hidden_size=768, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=12, image_size=224, patch_size=16),
        text_config=dict(vocab_size=32, hidden_size=32, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=4))
    original = transformers.SiglipModel(config).eval()
    weights = original.state_dict()
    vision_key = 'vision_model.embeddings.patch_embedding.weight'
    if damage == 'missing_vision':
        del weights[vision_key]
    elif damage == 'mismatched_vision':
        weights[vision_key] = weights[vision_key][:1].clone()
    elif damage == 'unexpected_vision':
        weights['vision_model.unexpected_probe'] = torch.zeros(1)
    original.save_pretrained(tmp_path, state_dict=weights)
    processor_class = getattr(transformers, 'SiglipImageProcessorPil', None) or transformers.SiglipImageProcessor
    expected_processor = processor_class(size={'height': 224, 'width': 224})
    expected_processor.save_pretrained(tmp_path)
    files = [tmp_path / name for name in ('config.json', 'preprocessor_config.json', 'model.safetensors')]
    manifest = dict(model_id=MODEL_ID, revision='local-fixture', files={p.name: file_sha256(p) for p in files})
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest))
    # Capture the real HF loading report before our strict validation. Expected
    # text/scoring keys should be absent; vision problems must still be visible.
    loading = {}
    original_loader = transformers.SiglipVisionModel.from_pretrained.__func__
    global_ignore_patterns = deepcopy(transformers.SiglipVisionModel._keys_to_ignore_on_load_unexpected)
    def capture_loader(cls, *args, **kwargs):
        model, report = original_loader(cls, *args, **kwargs)
        loading.update(report)
        return model, report
    monkeypatch.setattr(transformers.SiglipVisionModel, 'from_pretrained', classmethod(capture_loader))
    if damage:
        # A size mismatch can be rejected inside Transformers before our check.
        with pytest.raises((ValueError, RuntimeError), match='vision|size mismatch|ignore_mismatched_sizes'):
            FrozenSiglipPatchEncoder.from_local(tmp_path)
        if damage == 'missing_vision':
            assert vision_key in loading['missing_keys']
        elif damage == 'unexpected_vision':
            assert 'vision_model.unexpected_probe' in loading['unexpected_keys']
        return
    loaded = FrozenSiglipPatchEncoder.from_local(tmp_path)
    assert not loading['unexpected_keys'] and not loading['missing_keys']
    assert transformers.SiglipVisionModel._keys_to_ignore_on_load_unexpected == global_ignore_patterns
    images = torch.rand(1, 3, 256, 256) * 2 - 1
    pixels = loaded.processor(images=list(((images + 1) / 2).numpy()), do_rescale=False,
                              return_tensors='pt', input_data_format='channels_first')['pixel_values']
    expected_pixels = expected_processor(images=list(((images + 1) / 2).numpy()), do_rescale=False,
                                        return_tensors='pt', input_data_format='channels_first')['pixel_values']
    torch.testing.assert_close(pixels, expected_pixels, atol=0, rtol=0)
    with torch.no_grad():
        expected = original.vision_model(pixel_values=pixels).last_hidden_state
    torch.testing.assert_close(loaded(images), expected)
    assert not any('text_model' in name for name, _ in loaded.named_parameters())


def test_manifest_hash_verification_and_download_token_handling(tmp_path, monkeypatch):
    hub = pytest.importorskip('huggingface_hub')
    calls = {}
    def info(self, repo, **kwargs):
        calls['info'] = repo, kwargs
        return SimpleNamespace(sha='immutable-sha')
    def download(**kwargs):
        calls['download'] = kwargs
        (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'siglip'}))
        (tmp_path / 'preprocessor_config.json').write_text('{}')
        (tmp_path / 'model.safetensors').write_bytes(b'offline fixture')
    monkeypatch.setattr(hub.HfApi, 'model_info', info)
    monkeypatch.setattr(hub, 'snapshot_download', download)
    monkeypatch.setenv('HF_TOKEN', 'test-token')
    script = Path(__file__).resolve().parents[1] / 'scripts/download_siglip2.py'
    spec = importlib.util.spec_from_file_location('download_siglip2_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.download(tmp_path)
    assert manifest['revision'] == calls['download']['revision'] == 'immutable-sha'
    assert calls['download']['token'] == 'test-token'
    assert 'test-token' not in (tmp_path / MANIFEST_NAME).read_text()
    assert snapshot_identity(tmp_path) == manifest
    (tmp_path / 'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        snapshot_identity(tmp_path)


@pytest.fixture
def v2_training_setup(latent_training_setup, monkeypatch):
    import diffusion_ot.data.ground_truth as ground_truth
    import diffusion_ot.integrations.sit_diffusers as sit
    import diffusion_ot.models.pdae_sit as pdae
    run, config, latest, root, _, _, source = latent_training_setup
    config['train']['initialize_from'] = None
    config['refinement'] = {'enabled': False}
    config['encoder'] = dict(kind='siglip2_vit_b16', input_space='rgb', image_size=8,
                             local_dir='artifacts/siglip')
    config['adapter'] = dict(kind='pdae_v2_cross_attention', cross_attention_heads=2,
                             lora=True, lora_rank=2, lora_alpha=2)
    config['semantic_cfg'] = dict(enabled=True, dropout_probability=.1)
    torch.manual_seed(19)
    original, encoder = base_model(), TinyFrozenEncoder()
    original_vae = TinyVAE().requires_grad_(False)
    monkeypatch.setattr(ground_truth, 'load_afhq_dataset', lambda *a, **k: source)
    monkeypatch.setattr(FrozenSiglipPatchEncoder, 'from_local', lambda *a: deepcopy(encoder))
    def components(*a, **kw):
        vae = deepcopy(original_vae)
        latest['vae'] = vae
        return SimpleNamespace(transformer=deepcopy(original), vae=vae)
    def build(*a, **kw):
        latest['branch'] = build_pdae_sit_branch(*a, **kw)
        return latest['branch']
    monkeypatch.setattr(sit, 'load_sit_components', components)
    monkeypatch.setattr(pdae, 'build_pdae_sit_branch', build)
    return run, config, latest, root


def test_real_training_loop_validation_and_resume(v2_training_setup):
    run, _, latest, root = v2_training_setup
    first, logs, report = run('v2', steps=1)
    assert report.initial_step == 0 and report.final_step == 1
    assert first['model']['format_version'] == 6 and 'encoder' not in first['model']
    assert first['train_state']['refinement'] is None
    assert logs['refinement'] == []
    assert logs['train'][0]['encoder_grad_norm_pre_clip'] == 0
    assert logs['train'][0]['adapter_grad_norm_pre_clip'] > 0
    resumed, logs, report = run('v2', steps=2, resume=True)
    assert report.initial_step == 1 and report.final_step == 2
    assert resumed['ema']['num_updates'] == 2
    assert [row['step'] for row in logs['validation']] == [0, 1, 2]
    assert latest['branch'].training and not latest['branch'].encoder.training
    assert all(p.grad is None for p in latest['vae'].parameters())
    assert all(not name.startswith('encoder.') for name in resumed['ema']['shadow'])


def test_matched_augmentation_training_resume_and_validation_rng(v2_training_setup, monkeypatch, capsys):
    run, config, latest, _ = v2_training_setup
    first, _, _ = run('v2_aug_resume', steps=1)

    def encode(self, image):
        assert not torch.is_grad_enabled() and not self.training
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=torch.cat(
            [image, image.mean(1, keepdim=True)], 1)))
    monkeypatch.setattr(TinyVAE, 'encode', encode, raising=False)
    original = PDAEV2Branch.forward
    observed = []
    def forward(self, *args, **kwargs):
        if self.training:
            rgb = kwargs['encoder_image']
            torch.testing.assert_close(kwargs['x0_latent'], torch.cat([rgb, rgb.mean(1, keepdim=True)], 1))
            observed.append(rgb.detach().clone())
        return original(self, *args, **kwargs)
    monkeypatch.setattr(PDAEV2Branch, 'forward', forward)
    config['augmentation'] = dict(enabled=True, vae_batch_size=2,
        horizontal_flip_probability=1., color_jitter=dict(probability=1.), affine=dict(probability=0.))
    config['evaluation']['compare_raw_ema'] = True
    resumed, logs, report = run('v2_aug_resume', steps=2, resume=True)
    assert report.initial_step == 1 and resumed['ema']['num_updates'] == first['ema']['num_updates'] + 1
    assert resumed['train_state']['ema_reset'] is None
    assert resumed['train_state']['data_recipe_change']['step'] == 1
    assert 'data_recipe_changed_on_resume' in capsys.readouterr().out
    assert all(state['step'] == 2 for state in resumed['optimizer']['state'].values())
    assert len(observed) == 2  # Two accumulated microbatches; legacy flip was bypassed.
    row = logs['train'][-1]
    assert row['augmentation']['fractions']['horizontal_flip'] == 1.
    assert row['adapter_grad_norm_pre_clip'] > 0 and row['lora_grad_norm_pre_clip'] > 0
    assert row['loss_interval_mean'] == pytest.approx(row['loss'])
    assert all(p.grad is None for p in latest['vae'].parameters())
    assert all(p.grad is None for p in latest['branch'].encoder.parameters())
    paired, _, _ = run('v2_aug_paired', steps=2)
    single, _, _ = run('v2_aug_single', steps=2,
                       modify=lambda c: c['evaluation'].update(compare_raw_ema=False))
    torch.testing.assert_close(paired['rng_state'], single['rng_state'], atol=0, rtol=0)
    for name in paired['ema']['shadow']:
        torch.testing.assert_close(paired['ema']['shadow'][name], single['ema']['shadow'][name], atol=0, rtol=0)


def test_lr_warmup_accumulation_resume_and_legacy_schedule_guard(v2_training_setup):
    run, config, _, _ = v2_training_setup
    config['train'].update(lr_adapter=1e-4, lr_lora=2.5e-5, weight_decay=.01,
        lr_schedule=dict(type='constant_with_warmup', warmup_steps=4))
    # Fixture uses two microbatches/update; LR still advances only once/update.
    first, logs, _ = run('v2_warmup', steps=1)
    assert logs['train'][0]['lr_factor'] == .25
    assert logs['train'][0]['learning_rates'] == pytest.approx({'adapter': 2.5e-5, 'lora': 6.25e-6})
    assert all(group['weight_decay'] == .01 for group in first['optimizer']['param_groups'])
    resumed, logs, report = run('v2_warmup', steps=4, resume=True)
    assert report.initial_step == 1
    assert [row['lr_factor'] for row in logs['train']] == [.25, .5, .75, 1.]
    assert logs['train'][-1]['learning_rates'] == pytest.approx({'adapter': 1e-4, 'lora': 2.5e-5})
    assert resumed['train_state']['lr_schedule'] == config['train']['lr_schedule']
    with pytest.raises(ValueError, match='LR schedule changed on resume'):
        run('v2_warmup', steps=5, resume=True,
            modify=lambda c: c['train'].update(lr_schedule={'type': 'constant'}))
    run('v2_legacy_lr', steps=1, modify=lambda c: c['train'].pop('lr_schedule'))
    with pytest.raises(ValueError, match='LR schedule changed on resume'):
        run('v2_legacy_lr', steps=2, resume=True)


@pytest.mark.parametrize('primary_ema', [True, False])
def test_paired_validation_preserves_updates_and_uses_same_inputs(v2_training_setup, primary_ema):
    run, config, _, _ = v2_training_setup
    config['ema'].update(decay=.9, warmup_steps=0)
    config['evaluation']['compare_raw_ema'] = True
    config['evaluation']['use_ema'] = primary_ema
    paired, logs, _ = run('v2_paired', steps=2)
    single, _, _ = run('v2_single', steps=2,
                        modify=lambda c: c['evaluation'].update(compare_raw_ema=False))
    # The additional evaluation must not alter the following training update.
    for key in paired['ema']['shadow']:
        torch.testing.assert_close(paired['ema']['shadow'][key], single['ema']['shadow'][key], atol=0, rtol=0)
    torch.testing.assert_close(paired['rng_state'], single['rng_state'], atol=0, rtol=0)
    rows = logs['validation']
    assert all(row['sample_ids'] == rows[0]['sample_ids'] for row in rows)
    for row in rows:
        comparison = row['weight_comparison']
        assert row['num_samples'] == 4 and len(row['sample_ids']) == 4
        assert row['use_ema'] == primary_ema
        assert row['correct_z'] == comparison['ema' if primary_ema else 'raw']['correct_z']
        assert comparison['raw']['correct_z']['count'] == 4
        assert [b['count'] for b in comparison['raw']['time_bins']] == [b['count'] for b in row['time_bins']]
        assert comparison['ema_minus_raw_correct_z_mse'] == pytest.approx(
            comparison['ema']['correct_z']['mse'] - comparison['raw']['correct_z']['mse'])
        assert row['ema_settings']['decay'] == .9
    assert rows[0]['weight_comparison']['ema_minus_raw_correct_z_mse'] == 0
    with pytest.raises(ValueError, match='requires ema.enabled'):
        run('no_ema', steps=1, modify=lambda c: c['ema'].update(enabled=False))


def test_ema_decay_change_on_resume_preserves_history(v2_training_setup, capsys):
    run, config, latest, _ = v2_training_setup
    config['ema'].update(decay=.9, warmup_steps=0)
    first, _, _ = run('v2_ema_resume', steps=1)
    resumed, _, _ = run('v2_ema_resume', steps=2, resume=True,
                         modify=lambda c: c['ema'].update(decay=.8))
    assert resumed['ema']['decay'] == .8 and resumed['ema']['num_updates'] == 2
    raw = dict(latest['branch'].named_parameters())
    for name, value in resumed['ema']['shadow'].items():
        expected = first['ema']['shadow'][name] * .8 + raw[name].detach().cpu() * .2
        torch.testing.assert_close(value, expected)
    assert 'ema_settings_changed_on_resume' in capsys.readouterr().out


def test_real_evaluation_raw_ema_grids_and_roundtrip(v2_training_setup):
    import yaml
    from diffusion_ot.evaluation.stage1a_eval import run_stage1a_weight_comparison
    run, _, _, root = v2_training_setup
    checkpoint, _, _ = run('v2_eval', steps=1)
    train_path = root / 'v2_eval.yaml'
    updated_config = yaml.safe_load(train_path.read_text())
    updated_config['ema']['decay'] = .123  # The report must use the saved schedule.
    train_path.write_text(yaml.safe_dump(updated_config))
    config = dict(project_root=str(root), split='val', weights='ema',
        dataset=dict(smoke_samples=2, batch_size=2, num_workers=0),
        architecture=dict(require_encoder_kind='siglip2_vit_b16', require_encoder_input_space='rgb'),
        sampling=dict(smoke_num_steps=2, guidance_scales=[0., 1., 2.],
            variants=['correct_z', 'shuffled_z', 'null_z'],
            inferred_noise=dict(enabled=True, num_steps=2, variants=['correct_z'])),
        metrics=dict(image_reference='original_rgb'), input_statistics=dict(enabled=False))
    path = root / 'eval.yaml'
    path.write_text(yaml.safe_dump(config))
    comparison = run_stage1a_weight_comparison(root / 'v2_eval.yaml', path, device='cpu')
    report = SimpleNamespace(**comparison['reports']['ema'])
    raw = comparison['reports']['raw']
    assert raw['sample_ids'] == report.sample_ids and raw['seed'] == report.seed
    assert raw['row_order'] == report.row_order and raw['num_steps'] == report.num_steps
    assert Path(raw['grid_path']).is_file() and raw['grid_path'] != report.grid_path
    assert Path(comparison['comparison_path']).is_file()
    assert raw['metrics']['vae_reconstruction'] == report.metrics['vae_reconstruction']
    assert comparison['ema_minus_raw']['correct_z_cfg_1']['pixel_mse'] == pytest.approx(
        report.metrics['correct_z_cfg_1']['pixel_mse'] - raw['metrics']['correct_z_cfg_1']['pixel_mse'])
    assert report.checkpoint_step == 1 and report.weights == 'ema'
    assert report.checkpoint_ema['decay'] == checkpoint['ema']['decay'] != .123
    assert report.checkpoint_ema == raw['checkpoint_ema']
    assert report.roundtrip_enabled and raw['roundtrip_enabled']
    assert Path(report.grid_path).is_file()
    stats = json.loads(Path(report.extra_reports['condition_tokens']).read_text())
    assert stats['dim'] == 768 and stats['tokens_per_image'] == 4
    assert any('roundtrip' in key or 'inferred' in key for key in report.extra_reports)

    # An explicit skip must reach both passes even when the YAML enables inversion.
    without_roundtrip = run_stage1a_weight_comparison(train_path, path, device='cpu', roundtrip=False)
    for weights, saved_report in without_roundtrip['reports'].items():
        assert not saved_report['roundtrip_enabled']
        assert 'inferred_noise_roundtrip' not in saved_report['extra_reports']
        assert saved_report['metrics'] == comparison['reports'][weights]['metrics']


def test_reset_ema_on_resume_uses_raw_without_changing_training(v2_training_setup, capsys):
    import yaml
    from diffusion_ot.training.train_pdae_domain import train_pdae_domain

    run, config, latest, root = v2_training_setup
    config['ema'].update(decay=.9, warmup_steps=0)
    first, _, _ = run('v2_ema_reset', steps=2)
    before = {name: parameter.detach().clone() for name, parameter in latest['branch'].named_parameters()
              if parameter.requires_grad}
    source = root / 'saved_step_2.pt'
    torch.save(first, source)
    path = root / 'v2_ema_reset.yaml'
    config = yaml.safe_load(path.read_text())
    config['ema']['decay'] = .8
    config['evaluation'].update(compare_raw_ema=True, use_ema=False)
    path.write_text(yaml.safe_dump(config))

    normal = train_pdae_domain(path, max_steps=3, resume_from=source)
    normal_checkpoint = torch.load(normal.checkpoint_path, weights_only=False)
    normal_raw = {name: parameter.detach().clone() for name, parameter in latest['branch'].named_parameters()
                  if parameter.requires_grad}
    reset = train_pdae_domain(path, max_steps=3, resume_from=source, reset_ema_on_resume=True)
    saved = torch.load(reset.checkpoint_path, weights_only=False)
    assert reset.initial_step == 2 and reset.final_step == 3
    assert saved['ema']['num_updates'] == 3 and saved['ema']['decay'] == .8
    for name, parameter in latest['branch'].named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.detach(), normal_raw[name], atol=0, rtol=0)
            torch.testing.assert_close(saved['ema']['shadow'][name], before[name] * .8 + parameter.detach() * .2)
    assert any(not torch.equal(saved['ema']['shadow'][name], normal_checkpoint['ema']['shadow'][name])
               for name in before)
    torch.testing.assert_close(saved['rng_state'], normal_checkpoint['rng_state'], atol=0, rtol=0)
    assert all(state['step'] == 3 for state in saved['optimizer']['state'].values())
    event = saved['train_state']['ema_reset']
    assert event['step'] == 2 and event['weights'] == 'raw' and event['num_updates'] == 2
    assert 'ema_reset_from_raw_on_resume' in capsys.readouterr().out
    validation = [json.loads(line) for line in Path(reset.validation_log_path).read_text().splitlines()]
    assert validation[-1]['ema_settings']['last_reset'] == event

    # Ordinary later resumes preserve the new history; the flag is not sticky.
    followup = train_pdae_domain(path, max_steps=4, resume_from='latest')
    continued = torch.load(followup.checkpoint_path, weights_only=False)
    assert continued['train_state']['ema_reset'] == event
    for name, parameter in latest['branch'].named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(continued['ema']['shadow'][name],
                                       saved['ema']['shadow'][name] * .8 + parameter.detach() * .2)
    with pytest.raises(ValueError, match='requires a resume checkpoint'):
        train_pdae_domain(path, max_steps=5, reset_ema_on_resume=True)
    with pytest.raises(ValueError, match='greater than the checkpoint step'):
        train_pdae_domain(path, max_steps=4, resume_from='latest', reset_ema_on_resume=True)
    config['ema']['enabled'] = False
    config['evaluation']['compare_raw_ema'] = False
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='requires ema.enabled'):
        train_pdae_domain(path, resume_from='latest', reset_ema_on_resume=True)
