"""Resumable native Caltech clean-reference transfer, prior to watermarking.

The fixed 10,000-update default is a declared transfer budget, not a claim of
convergence or a reproduction of ImageNet pretraining. AdamW/constant 1e-4 are
retained from the pinned native training scripts. Raw (non-EMA) states compete
on the same validation images/noise; earliest minimum selects T. Main seeds
and protocol test images never participate. Importing this module is stdlib-only.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .storage import atomic_path, bind, identity, read_json, verify, write_json

CONDITIONS={'unet_caltech256':'guided_diffusion','sit_b_2_caltech256':'sit_b_2'}


def default_recipe(condition):
    kind=CONDITIONS[condition]
    return {'schema':'caltech_native_reference_recipe_v1',
        'objective':'native_denoising_loss' if kind=='guided_diffusion' else 'native_flow_matching_loss',
        'optimizer':{'name':'AdamW','lr':1e-4,'betas':[.9,.999],'eps':1e-8,'weight_decay':0.},
        'schedule':{'kind':'constant','original_horizon_updates':10000},
        'selection_unit':'optimizer_updates','max_steps':10000,'observe_every_steps':500,
        'checkpoint_every_steps':500,'development_seed':100,'batch_size':32,'microbatch_size':1,
        'validation_draws_per_image':1,'validation_seed':1000100,
        'precision':'float32','numerics':'deterministic_fp32_tf32_disabled','trainable_scope':'all_native_trainable_parameters',
        'selected_weights':'raw_no_ema','augmentation':'center_crop256_seeded_horizontal_flip',
        'validation_transform':'center_crop256_no_flip',
        'diffusion_timesteps':1000 if kind=='guided_diffusion' else None,
        'transport':'Linear_velocity_unweighted' if kind=='sit_b_2' else None,
        'vae_posterior':'sample_scaled_0.18215' if kind=='sit_b_2' else None,
        'conditioning_initialization':'new_caltech_embeddings_preserve_imagenet_null' if kind=='sit_b_2' else None,
        'class_dropout_probability':.1 if kind=='sit_b_2' else None,
        'budget_basis':'predeclared_Caltech_transfer_budget_not_pretraining_convergence_claim'}


def _recipe(condition, recipe):
    expected=default_recipe(condition)
    if not isinstance(recipe,dict) or set(recipe)!=set(expected):
        raise ValueError('explicit complete native reference recipe required; start from default_recipe')
    for field in ('schema','objective','development_seed','precision','numerics','trainable_scope','selected_weights',
                  'augmentation','validation_transform','diffusion_timesteps','transport','vae_posterior',
                  'conditioning_initialization','class_dropout_probability','selection_unit'):
        if recipe[field]!=expected[field]:raise ValueError('unsupported reference recipe '+field)
    for field in ('max_steps','observe_every_steps','checkpoint_every_steps','batch_size','microbatch_size','validation_draws_per_image','validation_seed'):
        if type(recipe[field]) is not int or recipe[field]<1:raise ValueError('positive integer required: '+field)
    if recipe['batch_size']%recipe['microbatch_size'] or recipe['max_steps']%recipe['observe_every_steps']:
        raise ValueError('microbatches and observation horizon must divide declared budgets exactly')
    schedule=recipe['schedule']
    if schedule.get('kind') not in ('constant','linear') or schedule.get('original_horizon_updates')!=recipe['max_steps']:
        raise ValueError('learning-rate horizon must remain the full predeclared observation budget')
    opt=recipe['optimizer']
    if opt.get('name')!='AdamW' or not 0<float(opt.get('lr',0)) or not 0<float(opt.get('eps',0)) or float(opt.get('weight_decay',-1))<0:
        raise ValueError('finite AdamW optimizer recipe required')
    if len(opt.get('betas',[]))!=2 or any(not 0<=float(v)<1 for v in opt['betas']):raise ValueError('AdamW betas invalid')
    if any(not math.isfinite(float(opt[k])) for k in ('lr','eps','weight_decay')):raise ValueError('nonfinite optimizer recipe')
    return deepcopy(recipe)


def _spec(source,condition):
    if isinstance(source,(str,Path)):source=read_json(source)
    spec=deepcopy(source.get('models',{}).get(condition) if 'models' in source else source)
    if condition not in CONDITIONS or not isinstance(spec,dict) or spec.get('kind')!=CONDITIONS[condition] or spec.get('dataset')!='caltech256':
        raise ValueError('reference preparation supports only declared Caltech U-Net/SiT conditions')
    return spec


def _members(spec):
    adaptation=spec.get('dataset_adaptation',{})
    manifests=spec.get('input_manifests',{})
    bindings={'train':adaptation.get('training_membership'),
        'validation':adaptation.get('validation_membership') or manifests.get('development_heldout'),
        'test':manifests.get('test') or manifests.get('main_heldout')}
    rows={};sets={}
    for role,binding in bindings.items():
        rows[role]=read_json(verify(binding))
        if not isinstance(rows[role],list) or not rows[role]:raise ValueError('empty '+role+' membership')
        if any(not isinstance(r,dict) or not {'path','sha256','bytes'}<=set(r) for r in rows[role]):
            raise ValueError('reference members require explicit image file bindings')
        sets[role]={r['sha256'] for r in rows[role]}
        if len(sets[role])!=len(rows[role]):raise ValueError('duplicate '+role+' image identities')
    if sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['validation']&sets['test']:
        raise ValueError('native adaptation train/validation/protocol test identities overlap')
    return bindings,rows


def reference_plan(source_config_or_model_spec, condition, recipe, output):
    """Read bounded manifests and plan identity; no torch, images, weights or writes."""
    spec=_spec(source_config_or_model_spec,condition);recipe=_recipe(condition,recipe)
    bindings,rows=_members(spec)
    initialization=spec.get('initialization_checkpoint')
    if not isinstance(initialization,dict) or not {'path','bytes','sha256'}<=set(initialization):
        raise ValueError('hash-bound initialization checkpoint required')
    p=Path(initialization['path'])
    if not p.is_file() or p.stat().st_size!=initialization['bytes']:raise ValueError('initialization checkpoint missing/size mismatch')
    if spec['kind']=='sit_b_2':
        from .generative_conditions import conditioning
        if conditioning(spec)!=(256,list(range(256)),256):raise ValueError('Caltech requires its own 256 labels and null label 256')
        classes=sorted({r.get('class_id') for r in rows['train']})
        if len(classes)!=256 or any(not isinstance(c,str) for c in classes):raise ValueError('all 256 Caltech classes must have explicit image class IDs')
        for role in ('validation','test'):
            if any(r.get('class_id') not in classes for r in rows[role]):raise ValueError('unknown Caltech class in '+role)
        declared=spec['conditioning'].get('class_ids',classes)
        if declared!=classes:raise ValueError('Caltech conditioning order must match the frozen sorted class IDs')
        spec['conditioning']['class_ids']=classes
        if spec.get('model_args',{}).get('num_classes')!=256:raise ValueError('SiT target constructor must have 256 classes')
    code={name:bind(Path(__file__).with_name(name))['sha256'] for name in
          ('generative_reference.py','generative_models.py','generative_conditions.py','rgb.py')}
    payload={'schema':'caltech_native_reference_identity_v1','condition':condition,'recipe':recipe,
        'initialization_checkpoint':initialization,
        'initialization_checkpoint_key':spec.get('initialization_checkpoint_key',spec.get('checkpoint_key')),
        'source':spec['source'],'model_args':spec['model_args'],'conditioning':spec.get('conditioning'),
        'vae':spec.get('vae'),'memberships':bindings,'downstream_input_manifests':spec['input_manifests'],'code':code}
    return {'schema':'caltech_native_reference_plan_v1','condition':condition,'identity':identity(payload),
        'identity_inputs':payload,'spec':spec,'recipe':recipe,'output':str(Path(output).resolve()),
        'memberships':bindings,'counts':{k:len(v) for k,v in rows.items()},
        'observation_steps':list(range(recipe['observe_every_steps'],recipe['max_steps']+1,recipe['observe_every_steps']))}


def learning_rate(recipe,completed_updates):
    horizon=recipe['schedule']['original_horizon_updates']
    if not 0<=completed_updates<horizon:raise ValueError('LR index outside original training horizon')
    factor=1. if recipe['schedule']['kind']=='constant' else 1.-completed_updates/horizon
    return recipe['optimizer']['lr']*factor


def _seed(*parts):
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:15],16)


def _indices(count,batch_size,step,seed):
    import torch
    start=step*batch_size;result=[]
    while len(result)<batch_size:
        epoch,offset=divmod(start+len(result),count)
        permutation=torch.randperm(count,generator=torch.Generator().manual_seed(_seed(seed,'epoch',epoch)))
        result.extend(permutation[offset:offset+batch_size-len(result)].tolist())
    return result


def _images(rows,device,flip_seeds=None):
    import torch
    from PIL import Image
    from .generative_models import verify_file
    from .rgb import center_crop
    xs=[]
    for i,row in enumerate(rows):
        with Image.open(verify_file(row)) as im:x=center_crop(im)
        if flip_seeds is not None and _seed(flip_seeds[i],'flip')%2:x=x.flip(-1)
        xs.append(x)
    return torch.stack(xs).to(device)


def native_loss(backend,images,labels,seeds):
    """Per-image native loss with explicit independent posterior/noise/time draws."""
    import torch
    generators=[torch.Generator(device='cpu').manual_seed(s) for s in seeds]
    if backend.kind=='guided_diffusion':
        t=torch.tensor([torch.randint(0,1000,(1,),generator=g).item() for g in generators],device=images.device)
        noise=torch.stack([torch.randn(images.shape[1:],generator=g) for g in generators]).to(images.device)
        return backend.task.training_losses(backend.model,images,t,noise=noise)['loss']
    if backend.kind!='sit_b_2':raise ValueError('unsupported native reference objective')
    with torch.no_grad():
        posterior=backend.load_decoder().encode(images).latent_dist
        eta=torch.stack([torch.randn(posterior.mean.shape[1:],generator=g) for g in generators]).to(images.device)
        latent=(posterior.mean+posterior.std*eta)*.18215
    noise=torch.stack([torch.randn(latent.shape[1:],generator=g) for g in generators]).to(images.device)
    t=torch.tensor([torch.rand((),generator=g).item() for g in generators],device=images.device)
    xt=(1-t[:,None,None,None])*noise+t[:,None,None,None]*latent
    output=backend.model(xt,t,torch.tensor(labels,device=images.device,dtype=torch.long))
    if output.shape!=latent.shape:raise ValueError('SiT native velocity shape differs from latent target')
    return ((output-(latent-noise))**2).flatten(1).mean(1)


def _labels(spec,rows):
    if spec['kind']=='guided_diffusion':return None
    lookup={c:i for i,c in enumerate(spec['conditioning']['class_ids'])}
    return [lookup[r['class_id']] for r in rows]


def _validate(backend,spec,rows,recipe):
    import torch
    training=backend.model.training;backend.model.eval();total=0.;count=0
    try:
        with torch.no_grad():
            for draw in range(recipe['validation_draws_per_image']):
                for offset in range(0,len(rows),recipe['microbatch_size']):
                    batch=rows[offset:offset+recipe['microbatch_size']]
                    seeds=[_seed(recipe['validation_seed'],r['sha256'],draw) for r in batch]
                    losses=native_loss(backend,_images(batch,backend.device),_labels(spec,batch),seeds)
                    if not bool(torch.isfinite(losses).all()):raise ValueError('nonfinite validation native loss')
                    total+=float(losses.double().sum());count+=losses.numel()
    finally:backend.model.train(training)
    return total/count


def _rng_state():
    import random
    import numpy as np
    import torch
    numpy=np.random.get_state()
    return {'python':random.getstate(),'numpy':[numpy[0],numpy[1].tolist(),numpy[2],numpy[3],numpy[4]],
        'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    import random
    import numpy as np
    import torch
    random.setstate(state['python']);n=state['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


def _save_tensor(path,value):
    import torch
    with atomic_path(path) as tmp:
        with tmp.open('wb') as stream:torch.save(value,stream)


def validate_reference(output,plan):
    """Revalidate committed artifacts before reporting reuse; no model loading."""
    from .generative_conditions import require_reference_contract
    output=Path(output);receipt=read_json(output/'reference_complete.json')
    if receipt.get('identity')!=plan['identity'] or receipt.get('status')!='complete':raise ValueError('reference completion identity mismatch')
    spec=read_json(verify(receipt['model_spec']))
    if spec['checkpoint']!=receipt['checkpoint'] or spec['dataset_adaptation']['T_evidence']!=receipt['T_evidence']:
        raise ValueError('reference committed artifacts disagree')
    verify(spec['checkpoint']);require_reference_contract(spec)
    evidence=read_json(verify(receipt['T_evidence']))
    if evidence.get('reference_identity')!=plan['identity'] or evidence.get('recipe_sha256')!=identity(plan['recipe']):
        raise ValueError('reference evidence recipe identity mismatch')
    return spec


def run_reference(source_config_or_model_spec,condition,recipe,output,device='cuda:0',*,stop_after_updates=None):
    """Finish/reuse native T selection; a stop limit only pauses an unchanged plan.

    Successful return is the finalized model spec. An intentional prefix pause
    returns ``status: incomplete`` plus the bound resume receipt. Exceptions
    leave the last atomic checkpoint available for exact scheduled-step replay.
    """
    plan=reference_plan(source_config_or_model_spec,condition,recipe,output)
    if (Path(output)/'reference_complete.json').exists():return validate_reference(output,plan)
    candidate=plan['spec']
    if candidate.get('status')=='complete' and candidate.get('dataset_adaptation',{}).get('T_evidence'):
        from .generative_conditions import require_reference_contract
        evidence_binding=candidate['dataset_adaptation']['T_evidence']
        evidence=read_json(verify(evidence_binding))
        if evidence.get('reference_identity')==plan['identity']:
            require_reference_contract(candidate);verify(candidate['checkpoint'])
            output_path=Path(output);output_path.mkdir(parents=True,exist_ok=True)
            plan_path=output_path/'reference_plan.json'
            if plan_path.exists() and read_json(plan_path)['identity']!=plan['identity']:raise ValueError('cannot adopt over changed reference identity')
            write_json(plan_path,plan)
            model_spec_path=output_path/'model_spec.json';write_json(model_spec_path,candidate)
            write_json(output_path/'reference_complete.json',{'schema':'caltech_native_reference_complete_v1',
                'status':'complete','identity':plan['identity'],'model_spec':bind(model_spec_path),
                'checkpoint':candidate['checkpoint'],'T_evidence':evidence_binding,'reuse':'existing_identical_reference_no_weight_copy'})
            return validate_reference(output,plan)
    import os
    previous_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if previous_workspace not in (None,':4096:8',':16:8'):
        raise ValueError('CUBLAS_WORKSPACE_CONFIG must support deterministic CUDA operations')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    previous=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic,torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32)
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        return _run_reference(source_config_or_model_spec,condition,recipe,output,device,stop_after_updates=stop_after_updates)
    finally:
        torch.use_deterministic_algorithms(previous[0],warn_only=previous[1])
        torch.backends.cudnn.benchmark=previous[2];torch.backends.cudnn.deterministic=previous[3]
        torch.backends.cudnn.allow_tf32=previous[4];torch.backends.cuda.matmul.allow_tf32=previous[5]
        if previous_workspace is None:os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)


def _run_reference(source_config_or_model_spec,condition,recipe,output,device,*,stop_after_updates):
    plan=reference_plan(source_config_or_model_spec,condition,recipe,output)
    output=Path(plan['output']);recipe=plan['recipe'];spec=plan['spec']
    if (output/'reference_complete.json').exists():return validate_reference(output,plan)
    if stop_after_updates is not None and (type(stop_after_updates) is not int or not 0<stop_after_updates<=recipe['max_steps']):
        raise ValueError('prefix stop must be within the unchanged original horizon')
    output.mkdir(parents=True,exist_ok=True)
    plan_path=output/'reference_plan.json'
    if plan_path.exists() and read_json(plan_path)['identity']!=plan['identity']:raise ValueError('cannot resume changed reference identity')
    if not plan_path.exists():write_json(plan_path,plan)
    import random
    import numpy as np
    import torch
    from .generative_models import load_reference_initialization
    _,rows=_members(spec)
    random.seed(recipe['development_seed']);np.random.seed(recipe['development_seed']);torch.manual_seed(recipe['development_seed'])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(recipe['development_seed'])
    backend=load_reference_initialization(spec,recipe,device)
    backend.model.requires_grad_(True).train()
    if backend.kind=='sit_b_2':backend.model.pos_embed.requires_grad_(False)
    opt=recipe['optimizer'];optimizer=torch.optim.AdamW(backend.model.parameters(),lr=opt['lr'],betas=tuple(opt['betas']),eps=opt['eps'],weight_decay=opt['weight_decay'])
    step=0;observations=[];best=None;previous_checkpoint=None
    resume_path=output/'resume.json'
    if resume_path.exists():
        receipt=read_json(resume_path)
        if receipt.get('identity')!=plan['identity']:raise ValueError('resume identity changed')
        previous_checkpoint=verify(receipt['checkpoint'])
        state=torch.load(previous_checkpoint,map_location='cpu',weights_only=True)
        if state['identity']!=plan['identity']:raise ValueError('resume tensor identity changed')
        backend.model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
        step=state['completed_updates'];observations=state['observations'];best=state['best'];_restore_rng(state['rng'])
        if receipt['completed_updates']!=step or not 0<=step<=recipe['max_steps']:raise ValueError('resume update counter mismatch')
        if best is not None:verify(best['checkpoint'])
        del state
    limit=recipe['max_steps'] if stop_after_updates is None else stop_after_updates
    if step>limit:raise ValueError('prefix stop precedes existing completed updates')
    while step<limit:
        indices=_indices(len(rows['train']),recipe['batch_size'],step,recipe['development_seed'])
        optimizer.zero_grad(set_to_none=True)
        lr=learning_rate(recipe,step)
        for group in optimizer.param_groups:group['lr']=lr
        for offset in range(0,len(indices),recipe['microbatch_size']):
            batch=[rows['train'][i] for i in indices[offset:offset+recipe['microbatch_size']]]
            seeds=[_seed(recipe['development_seed'],'train',step,offset+i,r['sha256']) for i,r in enumerate(batch)]
            with torch.autocast(device_type=backend.device.type,enabled=False):
                loss=native_loss(backend,_images(batch,backend.device,seeds),_labels(spec,batch),seeds).sum()/recipe['batch_size']
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite native training loss')
            loss.backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in backend.model.parameters()):raise ValueError('nonfinite native training gradient')
        optimizer.step();step+=1
        if step%recipe['observe_every_steps']==0:
            # Validation cannot advance training's model/dropout RNG sequence.
            rng=_rng_state()
            try:value=_validate(backend,spec,rows['validation'],recipe)
            finally:_restore_rng(rng)
            candidate=output/f'observation_{step:08d}.pt';_save_tensor(candidate,backend.model.state_dict());binding=bind(candidate)
            observations.append({'step':step,'validation_native_task_loss':value,'checkpoint_sha256':binding['sha256']})
            if best is None or value<best['loss']:
                best={'step':step,'loss':value,'checkpoint':binding}
                # Keep earlier best until the following resume commit succeeds.
            else:candidate.unlink()
        if step%recipe['checkpoint_every_steps']==0 or step%recipe['observe_every_steps']==0 or step==limit:
            target=output/f'resume_{step:08d}.pt'
            _save_tensor(target,{'identity':plan['identity'],'model':backend.model.state_dict(),'optimizer':optimizer.state_dict(),
                'completed_updates':step,'observations':observations,'best':best,'rng':_rng_state(),
                'original_horizon_updates':recipe['max_steps']})
            write_json(resume_path,{'schema':'caltech_native_reference_resume_v1','identity':plan['identity'],
                'completed_updates':step,'checkpoint':bind(target)})
            if previous_checkpoint is not None and previous_checkpoint!=target:previous_checkpoint.unlink(missing_ok=True)
            previous_checkpoint=target
            for old in output.glob('observation_*.pt'):
                if best is not None and old!=Path(best['checkpoint']['path']):old.unlink()
    if step<recipe['max_steps']:
        return {'status':'incomplete','identity':plan['identity'],'completed_updates':step,'original_horizon_updates':recipe['max_steps'],'resume':bind(resume_path)}
    if best is None or [r['step'] for r in observations]!=plan['observation_steps']:raise ValueError('incomplete T observations')
    import shutil
    checkpoint=output/'clean_selected.pt'
    with atomic_path(checkpoint) as tmp:shutil.copyfile(verify(best['checkpoint']),tmp)
    checkpoint_binding=bind(checkpoint)
    evidence={'schema':'generator_reference_selection_v1','target_dataset':'caltech256','checkpoint_sha256':checkpoint_binding['sha256'],
        'recipe_sha256':identity(recipe),'selection_rule':'earliest_minimum_validation_native_task_loss','selection_split':'validation',
        'test_used_for_selection':False,'frozen_before_main':True,'training_membership_sha256':plan['memberships']['train']['sha256'],
        'validation_membership_sha256':plan['memberships']['validation']['sha256'],'selected_step':best['step'],'observations':observations,
        'reference_identity':plan['identity'],'initialization_checkpoint':spec['initialization_checkpoint'],
        'validation_noise_policy':'fixed_per_image_hash_and_draw','selected_weights':'raw_no_ema','original_horizon_updates':recipe['max_steps']}
    evidence_path=output/'T_selection.json';write_json(evidence_path,evidence)
    result=deepcopy(spec);result['checkpoint']=checkpoint_binding;result['checkpoint_key']=None;result['status']='complete'
    result['dataset_adaptation']={'target_dataset':'caltech256','reference_kind':'controlled_adaptation','recipe':recipe,
        'selection_rule':'earliest_minimum_validation_native_task_loss','test_used_for_selection':False,
        'training_membership':plan['memberships']['train'],'validation_membership':plan['memberships']['validation'],
        'T_evidence':bind(evidence_path)}
    from .generative_conditions import require_reference_contract
    require_reference_contract(result)
    model_spec_path=output/'model_spec.json';write_json(model_spec_path,result)
    write_json(output/'reference_complete.json',{'schema':'caltech_native_reference_complete_v1','status':'complete','identity':plan['identity'],
        'model_spec':bind(model_spec_path),'checkpoint':checkpoint_binding,'T_evidence':bind(evidence_path)})
    return validate_reference(output,plan)
