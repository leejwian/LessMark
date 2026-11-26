"""Closed generator condition matrix, with explicit per-domain source assets."""
CONDITIONS = {
    'unet_imagenet': ('guided_diffusion', 'imagenet'),
    'sit_b_2_imagenet': ('sit_b_2', 'imagenet'),
    'styleswin256_ffhq': ('styleswin_256', 'ffhq'),
    'unet_caltech256': ('guided_diffusion', 'caltech256'),
    'sit_b_2_caltech256': ('sit_b_2', 'caltech256'),
}
KINDS = {key: value[0] for key, value in CONDITIONS.items()}


def condition_spec(source, condition):
    if condition not in CONDITIONS:
        raise ValueError('generator condition outside revised closed scope: '+str(condition))
    spec = source.get('models', {}).get(condition)
    kind,dataset=CONDITIONS[condition]
    if not isinstance(spec,dict) or spec.get('kind')!=kind or spec.get('dataset')!=dataset:
        raise ValueError('model kind/dataset does not match condition: '+condition)
    if spec.get('status') in ('pending','missing','unprepared') or not spec.get('checkpoint'):
        raise ValueError('missing clean generator checkpoint for condition: '+condition)
    if dataset=='caltech256':
        provenance=spec.get('dataset_adaptation')
        if not isinstance(provenance,dict) or provenance.get('target_dataset')!=dataset or not provenance.get('training_membership'):
            raise ValueError('Caltech256 checkpoint needs explicit adaptation and training-membership provenance')
    return spec


def conditioning(spec):
    entry=spec.get('conditioning')
    if not isinstance(entry,dict):
        raise ValueError('SiT conditioning must be explicitly declared')
    count=entry.get('num_classes'); allowed=entry.get('allowed_labels'); null=entry.get('unconditional_label')
    if type(count) is not int or count<1 or null!=count:
        raise ValueError('SiT class count and native null label must agree with checkpoint')
    if not isinstance(allowed,list) or not allowed or len(set(allowed))!=len(allowed) or any(type(v) is not int or not 0<=v<count for v in allowed):
        raise ValueError('SiT requires an explicit unique supported-label list')
    return count,allowed,null


def reference_contract_issues(spec, *, verify_bindings=False):
    """Validate a target-domain clean reference without substituting CE.

    Controlled evidence uses ``generator_reference_selection_v1`` with the
    bound recipe, memberships, observation ``step``/``validation_native_task_loss``
    pairs and the checkpoint hash at the earliest minimum. A reviewed task-ready
    release instead supplies ``generator_task_ready_reference_v1`` provenance.
    This function uses stdlib only and never loads weights or images.
    """
    if spec.get('dataset') != 'caltech256':
        return []
    import json
    import math
    from .storage import identity, verify, Blocked
    issues=[]
    def issue(field,reason,category='research_setting'):
        issues.append({'field':'dataset_adaptation.'+field,'category':category,'reason':reason})
    adaptation=spec.get('dataset_adaptation')
    if not isinstance(adaptation,dict):
        issue('recipe','explicit Caltech256 reference recipe or reviewed task-ready provenance is required')
        return issues
    if adaptation.get('target_dataset')!='caltech256':
        issue('target_dataset','target domain must be Caltech256','invalid_configuration')
    kind=adaptation.get('reference_kind','controlled_adaptation')
    if kind not in ('controlled_adaptation','reviewed_task_ready_checkpoint'):
        issue('reference_kind','unsupported target-domain reference policy','invalid_configuration')
        return issues
    training=adaptation.get('training_membership')
    validation=adaptation.get('validation_membership') or spec.get('input_manifests',{}).get('development_heldout')
    def bound(value,field):
        if not isinstance(value,dict) or not {'path','bytes','sha256'}<=set(value):
            issue(field,'explicit hash-bound '+field+' required','missing_evidence')
            return None
        if verify_bindings:
            try:return json.loads(verify(value).read_text())
            except (OSError,ValueError,KeyError,TypeError,Blocked) as exc:
                issue(field,'unavailable or changed '+field+': '+str(exc),'missing_evidence')
        return None
    training_rows=bound(training,'training_membership')
    validation_rows=bound(validation,'validation_membership')
    checkpoint_sha=(spec.get('checkpoint') or {}).get('sha256')
    recipe=adaptation.get('recipe')
    if kind=='controlled_adaptation':
        objectives={'guided_diffusion':{'native_denoising_loss','denoising_mse'},
                    'sit_b_2':{'native_flow_matching_loss','flow_matching_mse'}}
        if not isinstance(recipe,dict):
            issue('recipe','explicit native task objective, optimizer, schedule and observation budget are required before Caltech training')
            recipe={}
        else:
            if recipe.get('objective') not in objectives.get(spec.get('kind'),set()):
                issue('recipe.objective','declare the model-native denoising or flow-matching objective; classifier CE is not a generator objective')
            if not isinstance(recipe.get('optimizer'),dict) or not recipe['optimizer'].get('name'):
                issue('recipe.optimizer','explicit optimizer recipe required')
            if not isinstance(recipe.get('schedule'),dict):
                issue('recipe.schedule','explicit learning-rate schedule required')
            if recipe.get('selection_unit') not in ('optimizer_updates','epochs'):
                issue('recipe.selection_unit','declare optimizer_updates or epochs for T and the observation budget')
            for field in ('max_steps','observe_every_steps'):
                if type(recipe.get(field)) is not int or recipe[field]<1:
                    issue('recipe.'+field,'positive predeclared observation budget required')
            if type(recipe.get('development_seed')) is not int or recipe['development_seed']<0:
                issue('recipe.development_seed','development seed separate from main repetitions is required')
        if adaptation.get('selection_rule')!='earliest_minimum_validation_native_task_loss':
            issue('selection_rule','T must be the earliest minimum of validation native task loss')
        if adaptation.get('test_used_for_selection') is not False:
            issue('test_used_for_selection','explicitly exclude test data from T selection')
        evidence=bound(adaptation.get('T_evidence'),'T_evidence')
        if evidence is not None:
            expected={'schema':'generator_reference_selection_v1','target_dataset':'caltech256',
                      'checkpoint_sha256':checkpoint_sha,'recipe_sha256':identity(recipe),
                      'selection_rule':'earliest_minimum_validation_native_task_loss',
                      'selection_split':'validation','test_used_for_selection':False,'frozen_before_main':True,
                      'training_membership_sha256':training.get('sha256') if isinstance(training,dict) else None,
                      'validation_membership_sha256':validation.get('sha256') if isinstance(validation,dict) else None}
            if not isinstance(evidence,dict) or any(evidence.get(k)!=v for k,v in expected.items()):
                issue('T_evidence','T evidence does not bind the target checkpoint, recipe, memberships and validation-only selection','missing_evidence')
            else:
                observations=evidence.get('observations')
                valid=isinstance(observations,list) and bool(observations)
                if valid:
                    valid=all(isinstance(row,dict) and type(row.get('step')) is int and row['step']>0
                        and type(row.get('validation_native_task_loss')) in (int,float)
                        and math.isfinite(row['validation_native_task_loss'])
                        and isinstance(row.get('checkpoint_sha256'),str) and len(row['checkpoint_sha256'])==64
                        for row in observations)
                if valid:
                    steps=[r['step'] for r in observations]
                    interval,maximum=recipe.get('observe_every_steps'),recipe.get('max_steps')
                    expected_steps=(list(range(interval,maximum+1,interval)) if type(interval) is int and interval>0 and type(maximum) is int and maximum>0 else [])
                    valid=steps==expected_steps and bool(expected_steps)
                if not valid:
                    issue('T_evidence.observations','complete ordered observations for the declared development budget are required','missing_evidence')
                else:
                    selected=min(observations,key=lambda r:(r['validation_native_task_loss'],r['step']))
                    if evidence.get('selected_step')!=selected['step'] or checkpoint_sha!=selected['checkpoint_sha256']:
                        issue('T_evidence.selected_step','selected checkpoint is not the earliest minimum validation native loss','missing_evidence')
    else:
        evidence=bound(adaptation.get('provenance'),'provenance')
        if evidence is not None:
            expected={'schema':'generator_task_ready_reference_v1','target_dataset':'caltech256',
                      'checkpoint_sha256':checkpoint_sha,'provenance_reviewed':True,
                      'test_used_for_selection':False,'protocol_test_membership_excluded':True,
                      'training_membership_sha256':training.get('sha256') if isinstance(training,dict) else None,
                      'validation_membership_sha256':validation.get('sha256') if isinstance(validation,dict) else None}
            if not isinstance(evidence,dict) or any(evidence.get(k)!=v for k,v in expected.items()) or not str(evidence.get('source_url','')).startswith('https://') or not evidence.get('source_revision') or not evidence.get('native_training_recipe'):
                issue('provenance','reviewed task-ready provenance must bind domain, checkpoint, native recipe, memberships and source revision','missing_evidence')
    if verify_bindings:
        def membership(rows,field):
            if rows is None:return None
            if not isinstance(rows,list) or not rows:
                issue(field,'membership must be a nonempty image-identity list','missing_evidence');return None
            values=[]
            for row in rows:
                value=row.get('sha256') if isinstance(row,dict) else row
                if not isinstance(value,str) or not value:
                    issue(field,'membership contains an invalid image identity','missing_evidence');return None
                values.append(value)
            if len(values)!=len(set(values)):
                issue(field,'membership contains duplicate image identities','missing_evidence')
            return set(values)
        train_ids=membership(training_rows,'training_membership')
        val_ids=membership(validation_rows,'validation_membership')
        main_binding=(spec.get('input_manifests',{}).get('test')
                      or spec.get('input_manifests',{}).get('main_heldout'))
        main_ids=membership(bound(main_binding,'protocol_test_membership'),'protocol_test_membership')
        if train_ids is not None and val_ids is not None and train_ids&val_ids:
            issue('memberships','reference training and validation memberships overlap','missing_evidence')
        if main_ids is not None and ((train_ids is not None and main_ids&train_ids) or (val_ids is not None and main_ids&val_ids)):
            issue('memberships','reference training/selection used protocol held-out test images','missing_evidence')
    return issues


def require_reference_contract(spec):
    issues=reference_contract_issues(spec,verify_bindings=True)
    if issues:
        raise ValueError('Caltech256 clean reference is not validated: '+ '; '.join(item['reason'] for item in issues))
