"""CPU integration tests of the real Stage 1A loop with tiny model substitutes."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from test_decoded_translation import branch, TinyVAE, tiny_features
from test_infoot_semantic_prior import save_prior
from diffusion_ot.training.stage1a_refinement import (
    Stage1ARefinement, deterministic_branch, validate_refinement_config,
)
from diffusion_ot.training.train_pdae_domain import TrainableEMA, _initialize_stage1a_weights


@pytest.fixture
def setup(monkeypatch, tmp_path):
    import diffusion_ot.data.afhq as afhq
    import diffusion_ot.models.pdae_sit as pdae
    import diffusion_ot.integrations.sit_diffusers as sit
    import diffusion_ot.training.stage1a_refinement as refinement
    import diffusion_ot.training.train_pdae_domain as train

    torch.manual_seed(45)
    original = branch()
    vae, features = TinyVAE(), tiny_features()
    latest = {}
    def components(*a, **kw):
        value = deepcopy(original)
        latest.update(branch=value, vae=deepcopy(vae))
        return SimpleNamespace(transformer=value.semantic_transformer.base, vae=latest['vae'])
    monkeypatch.setattr(sit, "load_sit_components", components)
    monkeypatch.setattr(pdae, "build_pdae_sit_branch", lambda *a, **kw: latest['branch'])
    monkeypatch.setattr(refinement, "load_image_features", lambda *a, **kw: deepcopy(features))
    source = []
    (tmp_path / 'manifests').mkdir()
    for split, start, stop in [('train', 0, 8), ('val', 8, 12)]:
        records = []
        for i in range(start, stop):
            pixels = np.random.default_rng(i).integers(0, 256, (8, 8, 3), dtype=np.uint8)
            source.append({'image': Image.fromarray(pixels), 'label': 'cat'})
            record = dict(sample_id=f'afhq_cat_{i:06d}', domain='cat', hf_index=i,
                          image_column='image', dataset_id='test/AFHQ', dataset_split='train',
                          latent_shape=[4, 8, 8])
            records.append(record)
            target = tmp_path / 'latents' / f'cat_{split}' / (record['sample_id'] + '.pt')
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.randn(4, 8, 8, generator=torch.Generator().manual_seed(i)), target)
        (tmp_path / 'manifests' / f'cat_{split}.jsonl').write_text(
            ''.join(json.dumps(r) + '\n' for r in records))
    monkeypatch.setattr(afhq, "load_afhq_dataset", lambda *a, **kw: source)
    data_path = tmp_path / 'data.yaml'
    data_path.write_text(yaml.safe_dump(dict(project_root=str(tmp_path), dataset_id='test/AFHQ',
        split='train', image_size=8, center_crop=True, manifest_dir='manifests', latent_dir='latents')))
    (tmp_path / 'model.yaml').write_text('pretrained: pretrained.yaml\n')
    save_prior(tmp_path / 'prior.pt')
    ema = TrainableEMA(original, decay=.9)
    # Differentiate EMA initialization from raw initialization.
    for name, value in ema.shadow.items():
        if name.startswith('encoder.'):
            value.add_(.01)
    torch.save(dict(domain='cat', step=50000, model=original.pdae_state_dict(), ema=ema.state_dict()),
               tmp_path / 'source.pt')
    cfg = dict(project_root=str(tmp_path), domain='cat', data_config='data.yaml', model_config='model.yaml',
        device='cpu', train=dict(max_steps=3, seed=42, initialize_from='source.pt', initialization_weights='ema',
            lr_encoder=.001, lr_adapter=.001, lr_lora=.001, grad_clip_norm=.001,
            log_every=1, save_every=1, keep_step_checkpoints=True),
        dataloader=dict(batch_size=4, gradient_accumulation_steps=2, random_horizontal_flip=1.,
                        num_workers=0, pin_memory=False),
        ema=dict(enabled=True, decay=.9, warmup_steps=2),
        evaluation=dict(enabled=True, every_steps=1, batch_size=4, num_batches=1, num_time_bins=2),
        refinement=dict(enabled=True, feature_prior_path='prior.pt', batch_size=2, num_steps=2,
            every_steps=2, warmup_steps=2, validation_batch_size=2, validation_num_steps=3,
            perceptual_weight=.05, structure_weight=.025, code_contrastive_weight=.01,
            negative_similarity_threshold=.99999))

    def run(name, *, steps=3, resume=False, modify=None):
        config = deepcopy(cfg)
        config['output_dir'] = name
        if modify:
            modify(config)
        path = tmp_path / (name + '.yaml')
        path.write_text(yaml.safe_dump(config))
        report = train.train_pdae_domain(path, max_steps=steps, resume_from='latest' if resume else None)
        checkpoint = torch.load(report.checkpoint_path, weights_only=False)
        logs = {}
        for kind in ('train', 'validation', 'refinement'):
            file = tmp_path / name / 'logs' / (kind + '.jsonl')
            logs[kind] = [json.loads(line) for line in file.read_text().splitlines()] if file.exists() else []
        return checkpoint, logs, report

    return run, cfg, latest, tmp_path, features, vae, source


def test_refinement_trainer_warm_start_sparse_accumulation_validation_and_resume(setup):
    run, _, latest, root, _, _, _ = setup
    complete, logs, report = run('complete')
    assert report.initial_step == 0 and report.final_step == 3
    state = complete['train_state']
    assert state['initialization']['source_step'] == 50000
    assert state['initialization']['weights'] == 'ema'
    assert complete['ema']['num_updates'] == 3
    assert logs['validation'][0]['step'] == 0
    assert [r['step'] for r in logs['refinement']] == [1, 3]  # once/update, not once/microbatch
    assert logs['train'][1]['refinement'] is None
    for row in logs['refinement']:
        assert row['target'] == 'original_rgb'
        assert row['code_generator_grad_norm'] > 0
        assert row['applied_code_encoder_grad_norm'] == 0
        assert row['code_contrastive']['condition_bank_size'] == 4
        assert row['code_contrastive']['samples'] == 2
        assert row['weighted_loss'] == pytest.approx(row['weighted_image_loss'] + row['weighted_code_loss'])
    for row in logs['train']:
        assert row['loss'] == row['flow_loss']
        assert row['gradient_clipping'] == 'separate_encoder_generator'
    for row in logs['validation']:
        val = row['refinement']
        assert val['num_steps'] == 3 and val['ramp'] == 1.
        assert Path(val['grid_path']).is_file()
        assert val['grid_rows'][0] == 'original_rgb'
        assert val['sample_ids'] == logs['validation'][0]['refinement']['sample_ids']
    assert len(list((root / 'complete' / 'checkpoints').glob('step_*.pt'))) == 3
    run('resume', steps=1)
    resumed, resumed_logs, report = run('resume', resume=True)
    assert report.initial_step == 1 and report.final_step == 3
    assert [r['step'] for r in resumed_logs['refinement']] == [1, 3]
    torch.testing.assert_close(state['refinement']['noise_state'],
                               resumed['train_state']['refinement']['noise_state'], rtol=0, atol=0)
    assert resumed['ema']['num_updates'] == 3
    assert latest['branch'].training
    assert all(p.grad is None and not p.requires_grad for p in latest['vae'].parameters())


def test_code_recovery_changes_only_generator_updates_even_with_clipping(setup):
    run, _, _, _, _, _, _ = setup
    with_code, logs, _ = run('with_code', steps=1)
    without_code, _, _ = run('without_code', steps=1,
        modify=lambda c: c['refinement'].update(code_contrastive_weight=0.))
    assert logs['train'][0]['gradient_clip_fraction'] == 1.
    for name, value in with_code['model']['encoder'].items():
        torch.testing.assert_close(value, without_code['model']['encoder'][name], rtol=0, atol=0)
    before = without_code['model']['generator']['semantic_transformer']
    after = with_code['model']['generator']['semantic_transformer']
    def different(a, b):
        if isinstance(a, torch.Tensor):
            return not torch.equal(a, b)
        if isinstance(a, dict):
            return any(different(a[k], b[k]) for k in a)
        return False
    assert different(after, before)


def test_fresh_dino_only_training_skips_code_recovery_and_resumes(setup, monkeypatch):
    import diffusion_ot.training.stage1a_refinement as module
    run, _, _, _, _, _, _ = setup
    def unexpected_recovery(*args, **kwargs):
        raise AssertionError("Zero code weight must skip native-code recovery, including validation.")
    monkeypatch.setattr(module, 'generated_code_consistency_loss', unexpected_recovery)
    def dino_only(cfg):
        cfg['train'].update(initialize_from=None, lr_encoder=1e-4, lr_adapter=1e-4, lr_lora=2.5e-5)
        cfg['refinement'].update(perceptual_weight=.05, structure_weight=.01, code_contrastive_weight=0.)
    first, _, report = run('dino_only', steps=1, modify=dino_only)
    assert report.initial_step == 0
    assert first['train_state']['initialization'] is None
    resumed, logs, report = run('dino_only', steps=3, resume=True, modify=dino_only)
    assert report.initial_step == 1 and report.final_step == 3
    assert [r['step'] for r in logs['refinement']] == [1, 3]
    for row in logs['refinement'] + [r['refinement'] for r in logs['validation']]:
        assert row['code_contrastive'] is None
        assert row['code_gradient_routing'] == 'disabled'
        assert row['weighted_code_loss'] == 0
        assert row['weighted_loss'] == row['weighted_image_loss']
        assert row['weighted_image_loss'] == pytest.approx(
            row['ramp'] * (.05 * row['perceptual_loss'] + .01 * row['structure_loss']))
    for row in logs['refinement']:
        assert row['code_generator_grad_norm'] == row['applied_code_encoder_grad_norm'] == 0
    assert any(row['encoder_grad_norm_pre_clip'] > 0 for row in logs['train'])
    assert any(row['adapter_grad_norm_pre_clip'] > 0 for row in logs['train'])
    assert resumed['ema']['num_updates'] == 3
    assert not torch.equal(first['train_state']['refinement']['noise_state'],
                           resumed['train_state']['refinement']['noise_state'])


def test_resume_rejects_changed_loss_and_warm_start_checks_domain_and_ema(setup):
    run, _, latest, root, _, _, _ = setup
    run('guard', steps=1)
    with pytest.raises(ValueError, match='objective changed'):
        run('guard', resume=True, modify=lambda c: c['refinement'].update(code_temperature=.1))
    checkpoint = torch.load(root / 'source.pt', weights_only=False)
    with pytest.raises(ValueError, match='domain'):
        _initialize_stage1a_weights(latest['branch'], checkpoint, domain='dog', weights='ema')
    _initialize_stage1a_weights(latest['branch'], checkpoint, domain='cat', weights='ema')
    for name, p in latest['branch'].named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p, checkpoint['ema']['shadow'][name], rtol=0, atol=0)
    checkpoint['ema'] = None
    with pytest.raises(ValueError, match='EMA'):
        _initialize_stage1a_weights(latest['branch'], checkpoint, domain='cat', weights='ema')


def test_original_rgb_flip_alignment_and_validation_rng_isolation(setup):
    _, cfg, _, root, features, vae, _ = setup
    from diffusion_ot.data.latent_dataset import CachedLatentDataset, collate_latent_batch
    runtime = Stage1ARefinement(cfg, vae=vae, data_config_path=root / 'data.yaml', root=root, device='cpu', seed=42)
    ds = CachedLatentDataset(root / 'data.yaml', 'cat', project_root=root, random_horizontal_flip=1.)
    batch = collate_latent_batch([ds[i] for i in range(4)])
    value = branch()
    with deterministic_branch(value):
        a, b, metrics, (originals, _) = runtime.losses(value, value.semantic_transformer.base,
            batch, batch['x0_latent'], step=2)
        a.backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in value.encoder.parameters())
    assert any(p.grad is not None and p.grad.norm() > 0
               for name, p in value.named_parameters() if not name.startswith('encoder.'))
    from diffusion_ot.data.ground_truth import load_ground_truth_images
    expected = load_ground_truth_images(root / 'data.yaml', batch['metadata'][:2], dataset=runtime.original_dataset)
    torch.testing.assert_close(originals, expected.flip(-1))
    assert all(p.grad is None for p in runtime.features.parameters())
    assert all(p.grad is None for p in runtime.vae.parameters())
    saved = deepcopy(runtime.state_dict())
    rng = torch.get_rng_state().clone()
    loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=collate_latent_batch)
    a = runtime.evaluate(value, value.semantic_transformer.base, loader, device='cpu', dtype=torch.float32, step=1)
    b = runtime.evaluate(value, value.semantic_transformer.base, loader, device='cpu', dtype=torch.float32, step=2)
    assert a == b
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
    torch.testing.assert_close(saved['noise_state'], runtime.state_dict()['noise_state'], rtol=0, atol=0)
    assert value.training


def test_disabled_refinement_keeps_flow_only_training(setup):
    run, _, _, _, _, _, _ = setup
    checkpoint, logs, _ = run('legacy', steps=1, modify=lambda c: c.pop('refinement'))
    assert checkpoint['train_state']['refinement'] is None
    assert logs['refinement'] == []
    assert 'refinement' not in logs['validation'][0]
    assert logs['train'][0]['gradient_clipping'] == 'global'


def test_validation_does_not_change_training_updates(setup):
    run, _, _, _, _, _, _ = setup
    with_val, _, _ = run('with_validation', steps=2)
    without_val, _, _ = run('without_validation', steps=2,
                           modify=lambda c: c['evaluation'].update(enabled=False))
    def assert_equal(a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                assert_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                assert_equal(x, y)
        else:
            assert a == b
    for key in ('model', 'optimizer', 'ema', 'rng_state', 'dataloader_generator_state'):
        assert_equal(with_val[key], without_val[key])
    assert_equal(with_val['train_state']['refinement'], without_val['train_state']['refinement'])


@pytest.mark.parametrize('change', [dict(num_steps=0), dict(code_temperature=float('nan')),
                                   dict(negative_similarity_threshold=2), dict(perceptual_weight=-1)])
def test_invalid_refinement_options_fail_early(change):
    cfg = {'refinement': {'enabled': True, 'feature_prior_path': 'prior.pt', **change}}
    with pytest.raises(ValueError):
        validate_refinement_config(cfg)
