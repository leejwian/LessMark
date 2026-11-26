"""Paired RGB SSIM and real-distribution FID with explicit local provenance.

SSIM is channel-wise Gaussian-window population SSIM (11x11, sigma 1.5,
K1=.01, K2=.03, valid convolution, RGB data_range=1). FID uses local pinned
pytorch-fid Inception2048 weights and declared held-out real statistics.
"""
from __future__ import annotations
import json
import math
import fcntl
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .generative_models import verify_file
from .storage import bind, freeze_json, identity, write_json

SSIM_CONTRACT = {'window_size': 11, 'sigma': 1.5, 'k1': .01, 'k2': .03,
                 'data_range': 1., 'covariance': 'population', 'padding': 'valid',
                 'channels': 'RGB_equal_mean', 'dtype': 'float64'}
FID_PREPROCESSING = 'pytorch_fid_RGB01_resize299_no_quantization_v1'


def rgb_tensor(value):
    if not torch.is_tensor(value) or value.ndim != 4 or value.shape[1] != 3:
        raise ValueError('RGB must be a nonempty NCHW tensor with three channels')
    if not value.shape[0] or not torch.isfinite(value).all() or bool(((value < 0) | (value > 1)).any()):
        raise ValueError('RGB must contain finite values in [0,1]')
    return value


def rgb_ssim(reference, actual):
    x, y = rgb_tensor(reference).double(), rgb_tensor(actual).double()
    if x.shape != y.shape or min(x.shape[-2:]) < 11:
        raise ValueError('SSIM requires equal RGB shapes with spatial dimensions >=11')
    axis = torch.arange(-5, 6, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-axis.square()/(2*1.5**2)); kernel /= kernel.sum()
    window = torch.outer(kernel, kernel).expand(3, 1, 11, 11)
    mean = lambda z: F.conv2d(z, window, groups=3)
    mx, my = mean(x), mean(y)
    vx, vy = (mean(x*x)-mx*mx).clamp_min(0), (mean(y*y)-my*my).clamp_min(0)
    cov = mean(x*y)-mx*my
    score = ((2*mx*my+.01**2)*(2*cov+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
    return float(score.mean().clamp(-1, 1))


def frechet_distance(mu, covariance, real_mu, real_covariance):
    """PSD symmetric-root formulation, including rank-deficient small fixtures."""
    mu, real_mu = np.asarray(mu, dtype=np.float64), np.asarray(real_mu, dtype=np.float64)
    a, b = np.asarray(covariance, dtype=np.float64), np.asarray(real_covariance, dtype=np.float64)
    if mu.ndim != 1 or mu.shape != real_mu.shape or a.shape != (len(mu), len(mu)) or b.shape != a.shape:
        raise ValueError('FID moments have incompatible shapes')
    if not all(np.isfinite(v).all() for v in (mu, real_mu, a, b)):
        raise ValueError('FID moments must be finite')
    def psd(v):
        if not np.allclose(v, v.T, rtol=1e-8, atol=1e-10):
            raise ValueError('FID covariance must be symmetric')
        values, vectors = np.linalg.eigh((v+v.T)/2)
        if values.min() < -1e-8*max(1., np.abs(values).max()):
            raise ValueError('FID covariance must be positive semidefinite')
        return values.clip(min=0), vectors
    values, vectors = psd(a); psd(b)
    root = (vectors*np.sqrt(values)) @ vectors.T
    cross = root @ b @ root
    eigenvalues = np.linalg.eigvalsh((cross+cross.T)/2).clip(min=0)
    value = float(np.sum((mu-real_mu)**2)+np.trace(a)+np.trace(b)-2*np.sqrt(eigenvalues).sum())
    if value < -1e-7*max(1., float(np.trace(a)+np.trace(b))):
        raise ValueError('FID numerical result is unexpectedly negative')
    return max(value, 0.)


def fid_contract(settings, condition, dataset):
    if settings.get('required') is not True or type(settings.get('images')) is not int or settings['images'] < 2:
        raise ValueError('FID requires an explicit sample count >=2')
    if type(settings.get('batch')) is not int or settings['batch'] < 1:
        raise ValueError('FID requires a positive batch size')
    weights = settings.get('inception_weights'); verify_file(weights)
    reference = settings.get('real_statistics_by_condition', {}).get(condition)
    if not isinstance(reference, dict):
        raise ValueError('missing FID real statistics for condition: '+condition)
    metadata = json.loads(verify_file(reference['metadata']).read_text())
    stats = reference.get('statistics')
    bindings = stats if isinstance(stats, list) else [stats]
    if not bindings:
        raise ValueError('empty FID real statistics')
    split = settings.get('evaluation_split', 'heldout')
    expected = {'dataset': dataset, 'split': split, 'dims': 2048,
                'extractor_sha256': weights['sha256'], 'preprocessing': FID_PREPROCESSING}
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError('FID real dataset/split/extractor/preprocessing does not match condition')
    if type(metadata.get('samples')) is not int or metadata['samples'] < 2:
        raise ValueError('FID real sample count must be explicit')
    actual_hashes = [bind(verify_file(b))['sha256'] for b in bindings]
    if metadata.get('statistics_sha256') != actual_hashes or not metadata.get('membership_sha256'):
        raise ValueError('FID metadata must bind real statistic files and held-out membership')
    membership = None
    if split in ('validation','test'):
        membership = metadata.get('membership') or reference.get('membership')
        if not isinstance(membership, dict):
            raise ValueError('FID requires the actual bound real membership manifest')
        membership_path = verify_file(membership)
        members = json.loads(membership_path.read_text())
        if isinstance(members, dict):
            if members.get('split') != split or members.get('dataset') != dataset:
                raise ValueError('FID real membership must explicitly identify the declared dataset/split')
            members = members.get('entries')
        if not isinstance(members, list) or len(members) != metadata['samples']:
            raise ValueError('FID actual membership count differs from real statistics')
        identities = [r.get('sha256') if isinstance(r, dict) else r for r in members]
        if any(not isinstance(x, str) or len(x) != 64 for x in identities):
            raise ValueError('validation FID membership must contain SHA256 source identities')
        if len(set(identities)) != len(identities):
            # Existing official ImageNet statistics preserve 50,000 occurrences
            # with 49,971 distinct byte contents. Keep that actual membership;
            # do not silently deduplicate or relabel unrelated test moments.
            policy='official_validation_occurrence_preserving_v1'
            if metadata.get('validation_occurrence_policy') != policy:
                raise ValueError('duplicate validation content requires an explicit occurrence policy')
            audit_binding=metadata.get('validation_content_audit')
            receipt_binding=metadata.get('official_validation_source_receipt')
            audit=json.loads(verify_file(audit_binding).read_text())
            receipt=json.loads(verify_file(receipt_binding).read_text())
            paths=[r.get('path') if isinstance(r,dict) else None for r in members]
            if (len(set(paths))!=len(paths) or None in paths
                    or any(r.get('split')!='official_validation' for r in members)
                    or audit.get('policy')!=policy or audit.get('manifests',{}).get('fid_real')!=membership
                    or audit.get('group_counts',{}).get('fid_real')!=len(members)
                    or audit.get('unique_sha256_counts',{}).get('fid_real')!=len(set(identities))
                    or receipt.get('origin',{}).get('manifest')!=membership
                    or receipt.get('origin',{}).get('validation_occurrence_policy')!=policy
                    or receipt.get('origin',{}).get('weights')!=weights
                    or receipt.get('artifact',{}).get('sha256') not in actual_hashes
                    or receipt.get('real_images')!=len(members)
                    or receipt.get('unique_real_image_sha256')!=len(set(identities))):
                raise ValueError('official validation duplicate-content provenance does not match the real statistics')
        if metadata['membership_sha256'] != membership['sha256']:
            raise ValueError('FID statistics do not bind the submitted real membership')
        if split=='test':
            validation=settings.get('validation_membership_by_condition',{}).get(condition)
            if not isinstance(validation,dict):
                raise ValueError('final test FID requires actual bound selection-validation membership')
            if metadata.get('validation_membership')!=validation or metadata.get('validation_membership_sha256')!=validation['sha256']:
                raise ValueError('final test FID provenance does not bind the actual selection-validation membership')
            def identities(rows):
                if isinstance(rows,dict):rows=rows.get('entries')
                if not isinstance(rows,list) or not rows:raise ValueError('actual nonempty exclusion membership required')
                found=set()
                for row in rows:
                    if isinstance(row,dict):
                        if not isinstance(row.get('sha256'),str) or len(row['sha256'])!=64:
                            raise ValueError('invalid real-image exclusion SHA256 identity')
                        for key in ('sha256','content_sha256'):
                            if row.get(key):
                                if not isinstance(row[key],str) or len(row[key])!=64:raise ValueError('invalid exclusion content hash')
                                found.add(key+':'+row[key])
                    elif isinstance(row,str) and len(row)==64:found.add('sha256:'+row)
                    else:raise ValueError('invalid real-image exclusion identity')
                return found
            test_ids=identities(members)
            val_rows=json.loads(verify_file(validation).read_text())
            if test_ids&identities(val_rows):
                raise ValueError('final test FID real membership overlaps selection validation')
            training=settings.get('training_memberships_by_condition',{}).get(condition)
            if not isinstance(training,list) or dataset=='caltech256' and not training:
                raise ValueError('declare the known experiment-training/reference membership exclusions for final FID')
            if metadata.get('training_memberships')!=training:
                raise ValueError('test FID training-exclusion provenance differs from the actual experiment inputs')
            for member in training:
                if test_ids&identities(json.loads(verify_file(member).read_text())):
                    raise ValueError('final test FID real membership overlaps experiment training or carrier preparation')
    moments = []
    for b in bindings:
        with np.load(verify_file(b), allow_pickle=False) as data:
            mu, sigma = data['mu'].copy(), data['sigma'].copy()
        if mu.shape != (2048,) or sigma.shape != (2048,2048) or not np.isfinite(mu).all() or not np.isfinite(sigma).all():
            raise ValueError('FID real statistic dimensions or values invalid')
        moments.append((mu, sigma))
    return {'metadata': reference['metadata'], 'statistics': bindings, 'dataset': dataset,
            'real_samples': metadata['samples'], 'dims': 2048, 'preprocessing': FID_PREPROCESSING,
            'inception_weights': weights, **({'split': split, 'membership': membership,
                **{key:metadata[key] for key in ('validation_content_audit','official_validation_source_receipt','validation_occurrence_policy',
                    'validation_membership','validation_membership_sha256','training_memberships','official_dataset_split','test_usage_history') if key in metadata}}
                if split in ('validation','test') else {})}, moments


def load_fid_extractor(weights, device):
    import pytorch_fid.inception as inception
    path = verify_file(weights)
    original = inception.load_state_dict_from_url
    inception.load_state_dict_from_url = lambda *args, **kwargs: torch.load(path, map_location='cpu', weights_only=True)
    try:
        model = inception.InceptionV3([inception.InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    finally:
        inception.load_state_dict_from_url = original
    model.requires_grad_(False)
    return model


@torch.no_grad()
def generator_fid(backend, settings, condition, *, sampling, seeds, cache_dir, state_identity):
    """Generate exactly the declared FID sample set; cache moments by actual state."""
    contract, real = fid_contract(settings, condition, backend.spec['dataset'])
    if len(seeds) != settings['images'] or len(set(seeds)) != len(seeds):
        raise ValueError('FID seed count differs from frozen sample budget')
    origin = {'schema': 'generator_fid_origin_v1', 'condition': condition, 'state_identity': state_identity,
              'sampling': sampling, 'seeds': list(seeds), 'contract': contract,
              'implementation': bind(__file__)}
    root = Path(cache_dir)/identity(origin); root.mkdir(parents=True, exist_ok=True)
    with (root/'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _generator_fid_locked(backend,settings,condition,sampling,seeds,root,origin,contract,real)


def _generator_fid_locked(backend,settings,condition,sampling,seeds,root,origin,contract,real):
    # Shards may request an identical clean state. Only one process generates
    # and commits its statistics; followers verify and reuse that receipt.
    freeze_json(root/'origin.json', origin)
    if (root/'result.json').exists():
        record = json.loads((root/'result.json').read_text())
        if record['origin'] != bind(root/'origin.json'):
            raise ValueError('FID cache origin changed')
        verify_file(record['generated_statistics'])
        return record
    net = load_fid_extractor(settings['inception_weights'], backend.device)
    count, sums, products = 0, np.zeros(2048), np.zeros((2048,2048))
    for start in range(0,len(seeds),settings['batch']):
        batch = torch.cat([rgb_tensor(backend.sample_rgb(seed=int(seed), **sampling))
                           for seed in seeds[start:start+settings['batch']]], dim=0)
        features = net(batch)[0]
        features = F.adaptive_avg_pool2d(features,(1,1)).flatten(1).double().cpu().numpy()
        if len(features) != len(seeds[start:start+settings['batch']]) or not np.isfinite(features).all():
            raise ValueError('FID requires exactly one finite feature vector per declared seed')
        count += len(features); sums += features.sum(0); products += features.T @ features
    mu=sums/count; sigma=(products-count*np.outer(mu,mu))/(count-1)
    sigma=(sigma+sigma.T)/2
    temporary=root/'statistics.tmp.npz'; np.savez(temporary,mu=mu,sigma=sigma,n=count)
    temporary.replace(root/'statistics.npz')
    scores=[frechet_distance(mu,sigma,m,c) for m,c in real]
    record={'schema':'generator_real_fid_v1','condition':condition,'origin':bind(root/'origin.json'),
            'generated_statistics':bind(root/'statistics.npz'),'samples':count,
            'fid_values':scores,'fid_mean':float(np.mean(scores)), 'reference':contract,
            'evaluator':'pytorch-fid Inception2048','preprocessing':FID_PREPROCESSING}
    write_json(root/'result.json',record)
    return record


def paired_fid(clean, marked):
    if clean['samples'] != marked['samples'] or clean['reference'] != marked['reference']:
        raise ValueError('FID clean/watermarked must use same real reference and sample budget')
    if clean['condition'] != marked['condition']:
        raise ValueError('FID condition mismatch')
    return {'clean':clean,'watermarked':marked,
            'delta_fid':marked['fid_mean']-clean['fid_mean'],
            'delta_fid_values':[b-a for a,b in zip(clean['fid_values'],marked['fid_values'])]}
