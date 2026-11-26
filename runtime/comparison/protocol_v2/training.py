"""Update-based training with resumable fixed endpoints and legacy first-hit support."""
from __future__ import annotations
import copy
import math
import random
import uuid
from pathlib import Path
import numpy as np
import torch
from .storage import Blocked, bind, verify, identity, state_hash, read_json, write_json, save_torch, Cost


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"]:
        torch.cuda.set_rng_state_all(s["cuda"])


def optimizer_for(model, spec):
    kind = spec["name"]
    params = [p for p in model.parameters() if p.requires_grad]
    if kind == "SGD":
        return torch.optim.SGD(params, lr=spec["lr"], momentum=spec["momentum"], weight_decay=spec["weight_decay"], nesterov=spec.get("nesterov", False))
    if kind in ("Adam", "AdamW"):
        return getattr(torch.optim, kind)(params, lr=spec["lr"], betas=tuple(spec["betas"]), eps=spec["eps"], weight_decay=spec["weight_decay"])
    raise ValueError("optimizer must be explicitly SGD, Adam or AdamW")


def lr_at(update, spec):
    """Absolute 0-based update along original horizon; never compress to T."""
    base, horizon = spec["lr"], spec["horizon_updates"]
    if not 0 <= update < horizon:
        raise ValueError("requested update outside predeclared LR horizon")
    kind = spec["kind"]
    if kind == "constant":
        return base
    if kind == "multistep":
        return base*spec["gamma"]**sum(update >= u for u in spec["milestones_updates"])
    if kind == "warmup_cosine":
        warm = spec["warmup_updates"]
        if update < warm:
            return base*(update+1)/warm
        low = base*spec["min_lr_ratio"]
        return low+(base-low)*(1+math.cos(math.pi*(update-warm)/(horizon-warm)))/2
    if kind == "explicit":
        if len(spec["values"]) != horizon:
            raise ValueError("LR array/horizon mismatch")
        return spec["values"][update]
    raise ValueError("unknown LR schedule")


class UpdateBatches:
    """No worker prefetch: cursor + per-microbatch RNG is a reproducible boundary.

    Full effective batches only; epoch shuffle drops remainder. All paired
    branches use this same policy, including augmentation and dropout seed.
    """
    def __init__(self, dataset, microbatch, accumulation, seed):
        self.dataset, self.microbatch, self.accumulation, self.seed = dataset, microbatch, accumulation, seed
        self.effective_batch = microbatch*accumulation
        self.updates_per_epoch = len(dataset)//self.effective_batch
        if self.updates_per_epoch == 0:
            raise ValueError("dataset smaller than effective batch")
        self.epoch, self.order = None, None

    def get(self, update, micro):
        epoch, within = divmod(update, self.updates_per_epoch)
        if self.epoch != epoch:
            self.order = np.random.default_rng(self.seed+epoch).permutation(len(self.dataset))
            self.epoch = epoch
        start = within*self.effective_batch+micro*self.microbatch
        seed_all((self.seed*1000003+update*1009+micro) % (2**32))
        batch = [self.dataset[int(i)] for i in self.order[start:start+self.microbatch]]
        return torch.utils.data.default_collate(batch)


def checkpoint(path, model, optimizer, state, run_identity):
    path=Path(path)
    payload = {"identity": run_identity, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": state["scheduler"], "scaler": None, "rng": rng_state(), "data_cursor": {"next_update": state["updates"]}, "state": state}
    # Commit immutable blob first, then a single atomic pointer. A crash before
    # pointer commit leaves the previous complete checkpoint usable.
    previous=read_json(path) if path.exists() else None
    blob=path.parent/"checkpoint_states"/f"{path.stem}_{uuid.uuid4().hex}.pt"
    save_torch(blob,payload)
    pointer={"format":"atomic_checkpoint_pointer_v1","identity":run_identity,"artifact":bind(blob)}
    write_json(path,pointer)
    write_json(path.with_suffix(".receipt.json"),bind(path))
    # Only a superseded blob created for this mutable cursor is retired. Best
    # and observation checkpoints use separate blobs; historical assets untouched.
    if previous and previous.get("format")=="atomic_checkpoint_pointer_v1":
        old=Path(previous["artifact"]["path"])
        if old.parent==blob.parent and old.name.startswith(path.stem+"_"):
            old.unlink(missing_ok=True)


def load_checkpoint(path, run_identity=None):
    pointer=read_json(path)
    if pointer.get("format")!="atomic_checkpoint_pointer_v1":
        raise Blocked("resume_incompatible","checkpoint pointer format invalid")
    blob=verify(pointer["artifact"])
    # Trusted local checkpoint, hash verified before pickle deserialization.
    p = torch.load(blob, map_location="cpu", weights_only=False)
    if run_identity is not None and p["identity"] != run_identity:
        raise Blocked("resume_incompatible", f"checkpoint identity changed: {path}")
    return p


def resume_input_contract(contract,payload):
    """Accept the original starting model or the already-loaded saved endpoint.

    Endpoint callers still have to match every other input (keys, recipe,
    target, dataset/parent and optimizer origin). No arbitrary model is reused.
    """
    previous=payload["state"].get("input_contract")
    if previous and contract["initial_model_hash"]==state_hash(payload["model"]):
        contract={**contract,"initial_model_hash":previous["initial_model_hash"]}
    return contract


def train(model, batcher, task_loss, recipe, destination, run_identity, *, method=None, target=None, strength=0., deepmarks=False, device="cpu", smoke=False, budget=None, validation=None, observations=(), monitor=None, initial_optimizer=None, stop_at=None, cost_phase=None):
    """All task+watermark terms begin on the first actual optimizer update.

    validation must be task-only validation; test callbacks are not accepted
    for reference selection. T is selected only after the observation budget.
    """
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory / "latest.pt"
    declared_limit = recipe["updates"]
    fixed_endpoint = recipe.get("endpoint_policy") == "fixed_endpoint"
    limit = declared_limit if stop_at is None else stop_at
    if stop_at is not None and (not fixed_endpoint or not isinstance(stop_at,int) or not 0 < stop_at <= declared_limit):
        raise ValueError("stop_at requires a fixed endpoint within the immutable declared training budget")
    # Legacy DeepMarks first-hit behavior is disabled for the endpoint protocol.
    deepmarks = deepmarks and not fixed_endpoint
    if smoke and limit > (2 if recipe["phase"] == "ft" else 5):
        raise ValueError("smoke per-path optimizer update cap exceeded")
    if recipe["precision"] != "float32":
        raise Blocked("unsupported_configuration", "training supports explicit float32 only; no silent AMP change")
    if deepmarks and (method is None or method.name != "deepmarks" or recipe.get("initial_zero") not in ("accept", "require_one_update", "configuration_error")):
        raise Blocked("research_setting", "DeepMarks initial BER=0 policy must be fixed before training")
    if recipe["accumulation"] != batcher.accumulation:
        raise ValueError("accumulation contract mismatch")
    model.to(device)
    optimizer = optimizer_for(model, recipe["optimizer"])
    if recipe["optimizer_state"] == "inherit":
        if initial_optimizer is None and isinstance(recipe.get("initial_optimizer"),dict):
            source=recipe["initial_optimizer"]
            if source.get("kind")=="checkpoint":
                initial_optimizer=load_checkpoint(verify(source["artifact"]))["optimizer"]
        if initial_optimizer is None:
            raise Blocked("missing_asset", "inherit optimizer requested but reference optimizer unavailable")
        optimizer.load_state_dict(initial_optimizer)
    elif recipe["optimizer_state"] != "reset":
        raise ValueError("explicit optimizer reset/inherit rule required")
    # Bind actual starting model/key/optimizer bytes, not just their paths or
    # a method name. Resuming after changing any one must fail before updates.
    contract={"parent": run_identity, "recipe": recipe, "strength": strength, "method": method.name if method else "task_only", "target_hash":state_hash(target), "initial_model_hash":state_hash(model.state_dict()),"method_assets_hash":state_hash(method.assets) if method else None,"initial_optimizer_hash":state_hash(initial_optimizer),"batch_stream":{"seed":batcher.seed,"dataset_length":len(batcher.dataset),"microbatch":batcher.microbatch,"accumulation":batcher.accumulation}}
    saved=load_checkpoint(latest) if latest.exists() else None
    if saved: contract=resume_input_contract(contract,saved)
    run_identity=identity(contract)
    state = {"updates": 0, "updates_per_epoch":batcher.updates_per_epoch, "terminal": False, "reason": None, "history": [], "validation": [], "best_validation": None, "first_zero_update": None, "scheduler": {"spec": recipe["schedule"], "next_update": 0}, "monitor": None,"input_contract":contract}
    model.to(device)
    if latest.exists():
        p = saved
        if p["identity"]!=run_identity:
            raise Blocked("resume_incompatible",f"checkpoint inputs changed: {latest}")
        model.load_state_dict(p["model"])
        optimizer.load_state_dict(p["optimizer"])
        state = p["state"]
        restore_rng(p["rng"])
        if monitor and state.get("monitor"):
            monitor.load_state_dict(state["monitor"])
        if state["updates"] > limit:
            raise Blocked("resume_incompatible", "saved training cursor is beyond the requested endpoint")
        if state["terminal"] or state["updates"] == limit:
            # Preserve first hit forever; verify it instead of taking another step.
            if state["first_zero_update"] is not None:
                r = method.recover(model)
                if r.bits is None or not np.array_equal(r.bits, target):
                    raise RuntimeError("DeepMarks saved first-zero checkpoint failed reload verification")
            if validation is not None:
                _write_T(directory,state,declared_limit,recipe)
            write_json(directory/"history.json",state["history"])
            return state, latest
    elif deepmarks:
        r = method.recover(model)
        state["initial_ber"] = float(np.mean(r.bits != target)) if r.bits is not None else None
        if state["initial_ber"] == 0:
            if recipe["initial_zero"] == "configuration_error":
                raise Blocked("research_setting", "pre-insertion BER=0 encountered under predeclared error policy")
            if recipe["initial_zero"] == "accept":
                state.update(terminal=True, reason="initial_zero", first_zero_update=0)
                state["first_zero_signal"]=r.signal
                checkpoint(latest, model, optimizer, state, run_identity)
                _verify_first_zero(model, method, target, latest, run_identity)
                write_json(directory/"history.json",state["history"])
                return state, latest
    attempt = len(list(directory.glob("cost_*.json")))
    with Cost(cost_phase or recipe["phase"], device, output=directory/f"cost_{attempt:04d}.json") as cost:
        for update in range(state["updates"], limit):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            lr = lr_at(update, recipe["schedule"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            totals, nimages = {}, 0
            if monitor:
                monitor.before(model)
            if budget:
                budget.reserve("updates", 1, path=recipe["budget_path"], cap=2 if recipe["phase"] == "ft" else 5)
            for micro in range(recipe["accumulation"]):
                batch = batcher.get(update, micro)
                if budget:
                    budget.reserve("images", len(batch[0]))
                losses = task_loss(model, batch, device)
                cost.images += len(batch[0])
                wm = method.loss(model, target) if method else losses["total"].new_zeros(())
                objective = losses["total"]+strength*wm
                if not torch.isfinite(objective):
                    raise FloatingPointError("nonfinite training loss")
                (objective/recipe["accumulation"]).backward()
                for key,val in {**losses, "watermark": wm}.items():
                    totals[key] = totals.get(key, 0.)+float(val.detach())/recipe["accumulation"]
                nimages += len(batch[0])
            if recipe["gradient_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["gradient_clip"], error_if_nonfinite=True)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("nonfinite parameter gradient")
            if monitor:
                monitor.gradient(model, optimizer)
            optimizer.step()
            cost.updates += 1
            state["updates"] = update+1
            state["scheduler"]["next_update"] = update+1
            entry = {"update": update+1, "completed_epochs": (update+1)//batcher.updates_per_epoch, "update_within_epoch": (update+1)%batcher.updates_per_epoch, "updates_per_epoch":batcher.updates_per_epoch, "lr": lr, "images": nimages, "losses": totals}
            # MUST precede epoch/end-of-budget evaluation, persistence, or any other update.
            if deepmarks:
                r = method.recover(model)
                if r.bits is None:
                    raise FloatingPointError("DeepMarks update produced invalid coefficients")
                entry["native_ber"] = float(np.mean(r.bits != target))
                if entry["native_ber"] == 0:
                    state.update(terminal=True, reason="first_zero", first_zero_update=update+1)
                    state["first_zero_signal"]=r.signal
            if monitor:
                entry["dynamics"] = monitor.after(model)
                state["monitor"] = monitor.state_dict()
            state["history"].append(entry)
            if validation is not None and update+1 in observations:
                if getattr(validation, "split", None) != "validation":
                    raise ValueError("T selection cannot access test data")
                saved_rng = rng_state()
                metrics = validation(model)
                cost.images += metrics.get("images",0)
                restore_rng(saved_rng)
                if not math.isfinite(metrics["ce"]):
                    raise FloatingPointError("nonfinite validation CE")
                row = {"update": update+1, **metrics}
                state["validation"].append(row)
                if state["best_validation"] is None or row["ce"] < state["best_validation"]["ce"]:
                    state["best_validation"] = row
                    checkpoint(directory/"best_validation.pt", model, optimizer, state, run_identity)
            if update+1 == limit and not state["terminal"]:
                state.update(terminal=(limit == declared_limit), reason=("fixed_endpoint_pause" if limit < declared_limit else "budget_complete") if fixed_endpoint else (("smoke_budget_stop" if smoke else "ber_not_reached") if deepmarks else "budget_complete"))
            cadence=recipe.get("checkpoint_every_updates",1)
            if not isinstance(cadence,int) or cadence<1:
                raise ValueError("checkpoint cadence must be a positive update count")
            if state["terminal"] or update+1 == limit or update+1 in observations or (update+1)%cadence==0:
                checkpoint(latest, model, optimizer, state, run_identity)
            if update+1 in observations and recipe.get("retain_observation_checkpoints", True):
                checkpoint(directory/f"update_{update+1}.pt", model, optimizer, state, run_identity)
            if state["first_zero_update"] is not None:
                _verify_first_zero(model, method, target, latest, run_identity)
                break
    # Costs of previous attempts remain separate; no overwrite/double aggregation.
    if validation is not None:
        _write_T(directory,state,declared_limit,recipe)
    write_json(directory/"history.json",state["history"])
    return state, latest


def _write_T(directory,state,limit,recipe):
    if recipe.get("endpoint_policy") == "fixed_endpoint":
        if not state["terminal"] or state["updates"] != limit or not state["validation"] or state["validation"][-1]["update"] != limit:
            raise RuntimeError("fixed reference requires validation at its final declared checkpoint")
        write_json(directory/"T_selection.json", {"T":limit,"T_epochs":limit/state["updates_per_epoch"],"updates_per_epoch":state["updates_per_epoch"],"rule":"fixed budget last checkpoint; no validation checkpoint search","development_seed":100,"observations":state["validation"],"lr_horizon_preserved":recipe["schedule"],"checkpoint":bind(directory/"latest.pt")})
        return
    if not state["terminal"] or state["updates"] != limit or state["best_validation"] is None:
        raise RuntimeError("T requires the complete predeclared development observation window")
    write_json(directory/"T_selection.json", {"T": state["best_validation"]["update"], "T_epochs":state["best_validation"]["update"]/state["updates_per_epoch"], "updates_per_epoch":state["updates_per_epoch"], "rule": "earliest exact minimum validation CE", "development_seed": 100, "observations": state["validation"], "lr_horizon_preserved": recipe["schedule"], "checkpoint": bind(directory/"best_validation.pt")})


def _verify_first_zero(model, method, target, path, run_identity):
    p = load_checkpoint(path, run_identity)
    model.load_state_dict(p["model"])
    r = method.recover(model)
    if r.bits is None or not np.array_equal(r.bits, target):
        raise RuntimeError("DeepMarks first-zero save/reload mismatch; no further training allowed")


def classification_loss(model, batch, device):
    x,y = batch
    return {"total": torch.nn.functional.cross_entropy(model(x.to(device)), y.to(device))}
