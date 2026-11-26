import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100

# Launch from LERRA or supply its location.
def main(root, condition, output):
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(root))
    from experiment_2.code.classifier_lessmark import _backbone_readout, _patch_moment, insert_classifier_lessmark
    from experiment_2.code.input_paths import read_registered_json
    from runtime.comparison.protocol_v2.carriers import carriers, proportional_sample, q_vector
    from runtime.comparison.protocol_v2.data import build_model, transforms
    design = read_registered_json(root / 'experiment_2/config/NEXT_EXPERIMENT_DESIGN.json')
    state = design['embedding']['fixed_source_checkpoints'][condition]['checkpoint']
    raw = design['data']['source_CIFAR100']['source_files']
    scope = design['embedding']['lessmark_classifier_backbones']
    spec = scope['conditions'][condition]
    payload = design['payload']['single_watermark']
    bits = torch.tensor([int(x) for x in design['payload']['physical_payloads_by_recipient'][str(payload['recipient_id'])]])
    assert bits.shape == (256,) and sum(bits).item() == 128
    def digest_file(p):
        d = hashlib.sha256()
        with Path(p).open('rb') as f:
            for block in iter(lambda: f.read(4*1024*1024), b''): d.update(block)
        return d.hexdigest()
    def load_bound(b):
        p = root / b['path_from_project_root']
        assert digest_file(p) == b['sha256']
        return json.loads(p.read_text())
    split = load_bound(raw['validation_and_official_split'])
    membership = load_bound(raw['retained_membership_and_rgb_hashes'])
    assert membership['exact_rgb_content_disjoint']
    train_ids, val_ids = split['indices']['train'], split['indices']['validation']
    assert len(train_ids) == 44973 and len(val_ids) == 5000 and set(train_ids).isdisjoint(val_ids)
    assert digest_file(Path(raw['raw_data_root']) / raw['official_python_files'][0]['path_from_raw_data_root']) == raw['official_python_files'][0]['sha256']
    dataset = CIFAR100(raw['raw_data_root'], train=True, download=False, transform=transforms(split['preprocessing'],False))
    train_labels = [dataset.targets[i] for i in train_ids]
    samples = proportional_sample(train_labels, scope['calibration_images'], spec['sample_seed'])
    image_ids = [train_ids[i] for i in samples]
    assert len(image_ids) == len(set(image_ids)) == 16384
    assert digest_file(state['path']) == state['sha256']
    model_spec = {'architecture': 'resnet18' if condition.startswith('resnet') else 'mobilenet_v3_large', 'implementation': 'torchvision','classes':100,'cifar_stem':False}
    if model_spec['architecture'] == 'mobilenet_v3_large': model_spec['width_mult'] = 1.6
    clean = build_model(model_spec).eval()
    clean.load_state_dict(torch.load(state['path'],map_location='cpu',weights_only=True), strict=True)
    clean = clean.cuda()
    readout = _backbone_readout(clean, spec['weight_name'], spec['weight_shape'])
    def loader(indices):
        return DataLoader(Subset(dataset, indices), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    print('calibration started', condition, flush=True)
    moment, images, rows = _patch_moment(readout, loader(image_ids), scope['spatial_shape'])
    assert (images,rows)==(16384,802816)
    print('eigendecomposition', time.monotonic()-start,flush=True)
    k=scope['payload_bits']
    rng=20261125
    basis=carriers(moment.cpu(), [k], [rng])
    q=q_vector(readout.outputs,spec['q_seed'],classification=True)
    reference=readout.weight_matrix().cpu().double()
    alpha=0.2
    variants={}
    marked={}
    for kind in ['Low','Random0','High']:
        E=basis['bases'][f'{k}/{kind}'].contiguous()
        assets={'weight_name':spec['weight_name'],'weight_shape':spec['weight_shape'],'W0':reference,'E':E,'q':q,'classification':True,'tolerance':1e-10}
        model=copy.deepcopy(clean)
        recovery=insert_classifier_lessmark(model,assets,bits,alpha)
        direction=E @ (bits.double()*2-1)
        energy=(alpha**2 * (direction @ basis['moment'] @ direction)).item()
        eigsum= (alpha**2 * basis['eigenvalues'][:k].sum().item() if kind=='Low' else (alpha**2*basis['eigenvalues'][-k:].sum().item() if kind=='High' else None))
        if eigsum is not None:
            assert abs(energy-eigsum) <= max(1e-10,abs(energy)*1e-9)
        marked[kind]=model
        norm=(model.get_parameter(spec['weight_name']).detach().cpu().reshape(reference.shape).double()-reference).norm().item()
        variants[kind]={'actual_delta_weight_frobenius_norm':norm,'train_ideal_layer_mse':energy/readout.outputs,
                        'eigenvalue_sum_predicted_layer_mse':eigsum/readout.outputs if eigsum is not None else None,
                        'decode_status':recovery.status, 'bit_errors': int((torch.as_tensor(recovery.bits)!=bits).sum().item()),
                        'top1_correct':0, 'class_prediction_changed':0,'clean_correct_to_incorrect':0,'clean_incorrect_to_correct':0,
                        'local_sse':0.,'logit_sse':0.}
    print('validation started',time.monotonic()-start,flush=True)
    captures={}
    def hook_for(name):
        def capture(_module, _args, y):
            captures[name]=y.detach().clone()
        return capture
    all_models={'Clean':clean,**marked}
    path=spec['weight_name'].removesuffix('.weight')
    hooks=[model.get_submodule(path).register_forward_hook(hook_for(kind)) for kind,model in all_models.items()]
    clean_correct=0
    n=0
    for X,Y in loader(val_ids):
        X,Y=X.cuda(non_blocking=True),Y.cuda(non_blocking=True)
        clean_logits=clean(X)
        clean_pred=clean_logits.argmax(1)
        clean_ok=clean_pred==Y
        clean_correct+=int(clean_ok.sum())
        n+=len(X)
        for kind,model in marked.items():
            logits=model(X)
            pred=logits.argmax(1)
            ok=pred==Y
            metrics=variants[kind]
            metrics['top1_correct']+=int(ok.sum())
            metrics['class_prediction_changed']+=int((clean_pred!=pred).sum())
            metrics['clean_correct_to_incorrect']+=int((clean_ok&~ok).sum())
            metrics['clean_incorrect_to_correct']+=int((~clean_ok&ok).sum())
            metrics['local_sse']+=float((captures[kind].double()-captures['Clean'].double()).square().sum())
            metrics['logit_sse']+=float((logits.double()-clean_logits.double()).square().sum())
        captures.clear()
    for handle in hooks:handle.remove()
    assert n==len(val_ids)
    for x in variants.values():
        x['val_pre_bn_conv_mse']=x.pop('local_sse')/(n*scope['patches_per_image']*readout.outputs)
        x['val_logit_mse']=x.pop('logit_sse')/(n*100)
        x['val_top1_pct']=100*x['top1_correct']/n
        assert x['top1_correct']==clean_correct-x['clean_correct_to_incorrect']+x['clean_incorrect_to_correct']
    result={'scope':'C0 mechanism diagnostic, no source-task training or target-domain FT','condition':condition,'checkpoint_sha256':state['sha256'],
            'torch':torch.__version__,'torchvision':torchvision.__version__, 'gpu':torch.cuda.get_device_name(),
            'tf32':False,'deterministic_algorithms':True,'alpha':alpha,'random_orthonormal_basis_seed':rng,'payload_id':payload['recipient_id'],
            'train_calibration_images':images,'train_patches':rows,'sample_seed':spec['sample_seed'],'q_seed':spec['q_seed'],'val_images':n,
            'train_selected_official_id_sha256':hashlib.sha256(','.join(map(str,image_ids)).encode()).hexdigest(),
            'val_official_id_sha256':hashlib.sha256(','.join(map(str,val_ids)).encode()).hexdigest(),
            'clean_val_correct':clean_correct,'clean_val_top1_pct':100*clean_correct/n,'variants':variants,'elapsed_seconds':time.monotonic()-start}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('condition');parser.add_argument('output',type=Path)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    args=parser.parse_args()
    with torch.no_grad():main(args.root.resolve(), args.condition, args.output)
