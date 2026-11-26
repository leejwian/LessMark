"""Check the follow-up design, input weights and supporting historical records, read-only."""

import csv
import hashlib
import json
import sys
from decimal import Decimal
from itertools import combinations
from math import prod
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parent
CONFIG = EXPERIMENT / "config" / "NEXT_EXPERIMENT_DESIGN.json"
sys.path.insert(0, str(PROJECT))
from experiment_2.code.input_paths import read_registered_json as read_json


def verify_file(path, binding):
    path = Path(path)
    if "bytes" in binding:
        assert path.stat().st_size == binding["bytes"], path
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    assert digest.hexdigest() == binding["sha256"], path
    return path


def bound_config_path(binding):
    return verify_file(CONFIG.parent / binding["path_relative_to_this_file"], binding)


def read_bound_record(binding):
    return read_json(verify_file(binding["path"], binding))


def verify_cifar_membership(source):
    files = source["source_files"]
    records_file = files["retained_membership_and_rgb_hashes"]
    split_file = files["validation_and_official_split"]
    records = read_json(verify_file(PROJECT / records_file["path_from_project_root"], records_file))
    split = read_json(verify_file(PROJECT / split_file["path_from_project_root"], split_file))
    counts = source["counts"]
    assert counts == {"train": 44973, "validation": 5000, "official_test": 10000}
    content = {}
    for name in ("train", "validation", "test"):
        kept = [records["records"][name][i] for i in records["kept_positions"][name]]
        if name == "test":
            expected_ids = [f"cifar100/official_test/{i}" for i in range(counts["official_test"])]
        else:
            expected_ids = [split["record_ids"][i] for i in split["indices"][name]]
            assert len(expected_ids) == counts[name]
        assert [row["id"] for row in kept] == expected_ids, name
        content[name] = {row["content_sha256"] for row in kept}
    assert content["train"].isdisjoint(content["validation"] | content["test"])
    assert content["validation"].isdisjoint(content["test"])


def verify_photo_calibration(embedding):
    calibration = embedding["lessmark_real_image_calibration"]
    assert calibration["reference_checkpoint"] == "C_T"
    photos = calibration["photos_per_domain"]
    positions = calibration["positions_per_forward"]
    assert (photos, positions) == (1000, 64)
    lists = calibration["photo_lists"]
    assert set(lists) == {"imagenet", "caltech256", "ffhq"}
    for domain in ("imagenet", "caltech256"):
        entries = read_bound_record(lists[domain])
        assert len(entries) == len({entry["sha256"] for entry in entries}) == photos, domain
        for entry in entries:
            verify_file(entry["path"], entry)
        if domain == "imagenet":
            assert all(entry["split"] == "official_train" for entry in entries)
            assert len({entry["class_id"] for entry in entries}) == photos
        else:
            train = read_bound_record(lists[domain]["source_training_manifest"])
            assert {e["content_sha256"] for e in entries} <= {e["content_sha256"] for e in train}

    ffhq = lists["ffhq"]
    with bound_config_path(ffhq).open(encoding="ascii", newline="") as stream:
        rows = list(csv.DictReader(stream))
    train_ids = (f"{i:05d}" for i in range(60000))

    def order_key(image_id):
        return hashlib.sha256(f"lessmark-ffhq-1000-v1/{image_id}".encode("ascii")).digest()

    expected_ids = sorted(train_ids, key=order_key)[:photos]
    assert [row["image_id"] for row in rows] == expected_ids
    assert len({row["file_sha256"] for row in rows}) == photos
    for row in rows:
        path = Path(ffhq["image_directory"]) / (row["image_id"] + ".png")
        verify_file(path, {"sha256": row["file_sha256"]})

    conditions = calibration["conditions"]
    expected = {f"{model}_{domain}": (domain, 4) for model in ("unet", "sit_b_2")
                for domain in ("imagenet", "caltech256")}
    expected["styleswin256_ffhq"] = ("ffhq", 1)
    assert set(conditions) == set(expected)
    for name, (domain, draws) in expected.items():
        assert name in embedding["fixed_source_checkpoints"]
        condition = conditions[name]
        assert condition["photo_list"] == domain, name
        assert condition["feature_forwards_per_photo"] == draws, name
        assert condition["feature_rows"] == photos * draws * positions, name
    projection = calibration["styleswin_projection"]
    assert projection["trainable"] == "latent_z_only" and projection["latent_dimension"] == 512
    assert projection["runtime"] == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                                     "torch_deterministic_algorithms": True, "cudnn_benchmark": False}
    assert projection["optimizer"] == "Adam"
    assert projection["steps"] == 200  # One selection candidate per Adam update.
    assert projection["learning_rate"] == 0.05 and projection["betas"] == [0.9, 0.999]
    assert (projection["initial_seed_base"], projection["position_seed_base"]) == (100, 100100)


def verify_classifier_backbones(design):
    settings = design["embedding"]["lessmark_classifier_backbones"]
    assert settings["reference_checkpoint"] == "C_T" and settings["data_ref"] == "#/data/source_CIFAR100"
    assert settings["payload_bits"] == design["payload"]["message_length_bits"]
    assert settings["calibration_images"] == 16384 <= design["data"]["source_CIFAR100"]["counts"]["train"]
    assert settings["spatial_shape"] == [7, 7]
    assert prod(settings["spatial_shape"]) == settings["patches_per_image"] == 49
    assert settings["feature_rows"] == settings["calibration_images"] * settings["patches_per_image"]
    assert (EXPERIMENT / settings["implementation_from_experiment_root"]).is_file()

    expected = {
        "mobilenet_v3_large_w1p6_cifar100": ("features.15.block.3.0.weight", [256, 1536, 1, 1], 0, 3199773557, 1022235315),
        "resnet18_cifar100": ("layer4.1.conv2.weight", [512, 512, 3, 3], 1, 2458005299, 3855340113),
    }
    assert settings["conditions"].keys() == expected.keys()
    for condition, (name, shape, padding, sample_seed, q_seed) in expected.items():
        carrier = settings["conditions"][condition]
        assert (carrier["weight_name"], carrier["weight_shape"], carrier["padding"]) == (name, shape, [padding, padding])
        assert carrier["matrix_shape"] == [shape[0], prod(shape[1:])]
        assert prod(shape[1:]) >= 2 * settings["payload_bits"]
        assert (carrier["sample_seed"], carrier["q_seed"]) == (sample_seed, q_seed)


def verify_native_train_pools(pools, calibration):
    assert set(pools) == {"imagenet", "caltech256", "ffhq"}
    for dataset, count, represented, full in (("imagenet", 25000, 500, 1000), ("caltech256", 20713, 256, 256)):
        pool = pools[dataset]
        images = read_bound_record(pool["manifest"])
        assert len(images) == len({row["sha256"] for row in images}) == pool["available_images"] == count
        labels = {row["class_id"] for row in images}
        assert len(labels) == pool["represented_classes"] == represented
        assert pool["source_class_count"] == full
        if dataset == "imagenet":
            assert all(row["split"] == "official_train" for row in images)
            all_labels = {r["class_id"] for r in read_bound_record(calibration[dataset])}
            assert labels <= all_labels and len(all_labels) == full
        else:
            manifest = pool["manifest"]
            assert manifest == calibration[dataset]["source_training_manifest"]
            splits = read_json(Path(manifest["path"]).with_name("result.json"))["manifests"]
            assert manifest == splits["reference_train"]
            train_hashes = {row["content_sha256"] for row in images}
            assert len(train_hashes) == count
            for split, expected_size in (("reference_validation", 4351), ("main_heldout", 4699)):
                heldout = read_bound_record(splits[split])
                assert len(heldout) == expected_size
                assert train_hashes.isdisjoint(row["content_sha256"] for row in heldout), split
    ffhq = pools["ffhq"]
    assert ffhq["image_directory"] == calibration["ffhq"]["image_directory"]
    assert ffhq["official_training_ids_inclusive"] == [0, 59999] and ffhq["available_images"] == 60000
    required = {f"{i:05d}.png" for i in range(60000)}
    assert required <= {p.name for p in Path(ffhq["image_directory"]).glob("*.png")}


def verify_native_training(design, grids):
    embedding = design["embedding"]
    native = embedding["native_generator_training"]
    ft = design["fine_tuning"]
    assert native["objective"] == "native_source_task"
    assert native["use_clean_outputs_as_training_targets"] is False
    assert ft["use_clean_outputs_as_training_targets"] is False
    assert ft["watermark_loss_enabled"] is False
    pairing = native["pairing"]
    assert pairing["native_task_in_both_arms"] and pairing["independent_weights_and_optimizer_states"]
    assert pairing["watermark_from_generator_update"] == 1 and pairing["watermark_in_clean"] is False

    conditions = native["conditions"]
    expected = {f"{arch}_{domain}": (arch, domain) for arch in ("unet", "sit_b_2")
                for domain in ("imagenet", "caltech256")}
    expected["styleswin256_ffhq"] = ("styleswin", "ffhq")
    assert set(conditions) == set(embedding["lessmark_real_image_calibration"]["conditions"])
    assert {k: (v["architecture"], v["train_pool"]) for k, v in conditions.items()} == expected
    affected = {(name, method) for name in conditions for method in embedding["trained_methods"]}
    assert len(affected) == 20 and affected <= grids.keys()

    verify_native_train_pools(native["train_pools"], embedding["lessmark_real_image_calibration"]["photo_lists"])
    tasks = native["tasks"]
    families = {"unet": "epsilon_prediction_with_learned_variance", "sit_b_2": "linear_path_velocity",
                "styleswin": "adversarial_gan"}
    assert set(tasks) == set(families)
    for arch, family in families.items():
        target = ft["models"][arch]
        assert target["loss_family"] == tasks[arch]["loss"] == family
        assert target["dataset"] == "target_ArtBench"
        assert target["loss_terms_ref"] == f"#/embedding/native_generator_training/tasks/{arch}/loss_terms"
        assert "loss_terms" not in target, f"{arch} FT must use the shared source loss terms"
    assert tasks["unet"]["conditioning"] == "unconditional"
    assert tasks["unet"]["loss_terms"] == {
        "prediction": "mse_to_sampled_epsilon",
        "variance": "variational_bound_for_learned_reverse_variance",
        "weighting": "native_guided_diffusion_training_losses",
    }
    assert tasks["sit_b_2"]["loss_terms"] == {
        "path": "(1-t)*epsilon + t*z", "prediction": "mse_to_velocity", "velocity": "z - epsilon",
    }
    for sit in (tasks["sit_b_2"], ft["models"]["sit_b_2"]):
        assert sit["velocity_target"] == "z - epsilon" and sit["vae_trainable"] is False
    assert tasks["sit_b_2"]["conditioning"] == "original_source_dataset_class"
    for gan in (tasks["styleswin"], ft["models"]["styleswin"]):
        assert gan["discriminator_initialization"] == "identical_fresh_copies"
        assert gan["discriminator_shared_between_arms"] is False and gan["update_unit"] == "generator_optimizer_step"
    source_gan = tasks["styleswin"]
    assert source_gan["discriminator_watermark_loss"] is False
    assert source_gan["latent_sampling"] == {
        "coordinates": "independent_standard_normal",
        "training_draws": "fresh_batch_for_each_discriminator_and_generator_phase",
        "comparison_arms": "identical_latent_vectors_at_corresponding_phases",
    }
    target_gan = ft["models"]["styleswin"]
    sampling_ref = "#/embedding/native_generator_training/tasks/styleswin/latent_sampling"
    assert target_gan["latent_sampling_ref"] == sampling_ref and "latent_sampling" not in target_gan
    gan_terms = source_gan["loss_terms"]
    assert gan_terms["generator"] == "non_saturating_logistic"
    assert gan_terms["discriminator"] == "real_fake_logistic"
    assert gan_terms["r1"] == {
        "penalty": "squared_discriminator_input_gradient_on_real_images",
        "gamma": 10,
        "interval_discriminator_iterations": 16,
        "first_iteration_zero_based": 0,
        "lazy_step_weight": "gamma * interval / 2",
    }
    assert gan_terms["bcr"] == {
        "penalty": "mse_between_discriminator_scores_before_and_after_augmentation",
        "real_weight": 10, "fake_weight": 10,
    }
    assert source_gan["latent_dimension"] == 512 and source_gan["evaluated_weights"] == "actual_trained_generator"
    assert source_gan["checkpoint_key_at_start"] == embedding["fixed_source_checkpoints"]["styleswin256_ffhq"]["checkpoint_key"]

    budget = ft["generator_budget"]
    assert budget["unit"] == "generator_optimizer_step" and budget["shared_within"] == "source_model_and_dataset"
    assert budget["checkpoint"] == "fixed_common_T_FT"
    validation = ft["generator_validation"]
    assert validation["dataset"] == "target_ArtBench" and validation["updates_on_validation"] is False
    assert validation["real_images"] == design["data"]["target_ArtBench"]["counts"]["validation"]
    assert validation["native_diagnostics"].keys() == families.keys()
    assert set(design["generation_evaluation"]["matched_input_by_architecture"]) == set(families)
    return len(affected)


def verify_strength_grid(design, pristine_models):
    strength = design["strength"]
    canonical = read_json(PROJECT / "tools" / "method_names.json")["comparison_methods"]
    names = {method: data["display_name"] for method, data in canonical.items()}
    assert names == design["method_display_names"]
    baselines = design["embedding"]["fixed_source_checkpoints"]
    assert {(condition, method) for condition in baselines for method in names} == set(pristine_models)
    assert len(baselines) == strength["model_dataset_count"] == 7
    assert len(names) == strength["method_count"] == 6
    assert len({condition.rsplit("_", 1)[0] for condition in baselines}) == strength["architecture_count"] == 5
    assert strength["separate_center_search"] is False
    paper_strengths = strength["paper_strength_by_method"]
    assert set(paper_strengths) == set(names) - {"lerra"}
    assert paper_strengths == {
        "uchida": 0.01, "greedy": 0.01, "neuralmark": 1, "deepmarks": 0.1, "tattooed": 0.0009,
    }
    lessmark = strength["lessmark_alpha_by_model"]
    assert lessmark == {
        "mobilenet_v3_large_w1p6_cifar100": 0.2, "resnet18_cifar100": 0.2,
        "unet_imagenet": 0.1, "unet_caltech256": 0.1,
        "sit_b_2_imagenet": 1, "sit_b_2_caltech256": 1, "styleswin256_ffhq": 1,
    }
    assert lessmark.keys() == baselines.keys()
    for condition, baseline in baselines.items():
        contract = read_bound_record(pristine_models[condition, "lerra"]["contract"])
        original = (contract["source"]["original_clean"] if condition.endswith("_cifar100")
                    else contract["model"]["checkpoint"])
        checkpoint = baseline["checkpoint"]
        for key in ("path", "bytes", "sha256"):
            assert checkpoint[key] == original[key], (condition, key)
        verify_file(checkpoint["path"], checkpoint)

    multipliers = [Decimal(str(x)) for x in strength["multipliers"]]
    assert multipliers == [Decimal("0.25"), Decimal("0.5"), 1, 2, 4]
    columns = ["s0_div_4", "s0_div_2", "s0", "s0_x_2", "s0_x_4"]
    assert strength["candidate_columns_ascending"] == columns
    grids = {}
    table = bound_config_path(strength["candidate_table"])
    with table.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            condition, method = row["model_dataset"], row["method"]
            key = condition, method
            assert key not in grids, key
            assert row["display_method"] == names[method], key
            if method == "lerra":
                assert row["center_source"] == "prespecified_architecture", key
                center = Decimal(str(lessmark[condition]))
            else:
                assert row["center_source"] == "original_paper", key
                center = Decimal(str(paper_strengths[method]))
            values = [Decimal(row[col]) for col in columns]
            assert values == [center * factor for factor in multipliers], key
            grids[key] = values
    assert set(grids) == set(pristine_models)
    assert len(grids) == strength["combination_count"] == 42
    assert len(multipliers) == strength["candidates_per_combination"] == 5
    assert sum(map(len, grids.values())) == strength["combination_strength_count"] == 210
    return grids


def verify_training_recipe(design):
    recipe = read_json(bound_config_path(design["strength"]["campaign_training"]))
    assert set(recipe) == {"shared_arm_seed", "classification", "unet", "sit_b_2", "styleswin"}
    classifier = recipe["classification"]
    effective = classifier["microbatch"] * classifier["accumulation"]
    schedule = design["checkpoint_schedule"]
    assert effective == 64 and design["data"]["source_CIFAR100"]["counts"]["train"] // effective == 702
    assert schedule["classification_terminal_updates"] == 702 * 200
    assert classifier["optimizer"]["name"] == "SGD" and classifier["schedule"]["milestones_epochs"] == [100, 150]
    for family in ("sit_b_2", "unet", "styleswin"):
        params = recipe[family]
        assert params["batch_size"] == 32 and params["batch_size"] % params["microbatch_size"] == 0
        assert params["precision"] == "float32" and params["selected_weights"] == "raw"
        assert params["schedule"] == "constant" and params["optimizer"] == ("Adam" if family == "styleswin" else "AdamW")
    gan = design["embedding"]["native_generator_training"]["tasks"]["styleswin"]["loss_terms"]
    assert recipe["styleswin"]["r1_every"] == gan["r1"]["interval_discriminator_iterations"]
    assert recipe["styleswin"]["r1_gamma"] == gan["r1"]["gamma"]
    assert recipe["styleswin"]["bcr_real"] == gan["bcr"]["real_weight"]
    assert recipe["styleswin"]["bcr_fake"] == gan["bcr"]["fake_weight"]
    spreading = design["embedding"]["tattooed_spreading_length"]
    assert spreading["ratio"] == 1 / 800 and set(spreading["conditions"]) == set(design["embedding"]["fixed_source_checkpoints"])
    assert all(0 < r["spreading_length"] == r["eligible_coordinates"] // 800 for r in spreading["conditions"].values())


def verify_historical_checkpoints(design, pristine_models):
    schedule = design["checkpoint_schedule"]
    assert schedule["classification_epochs"] == [200] and schedule["classification_updates_per_epoch"] == 702
    assert schedule["generation_updates"] == [2048] and schedule["direct_additional_optimizer_updates"] == 0
    class_cap = schedule["classification_epochs"][-1] * schedule["classification_updates_per_epoch"]
    assert class_cap == schedule["classification_terminal_updates"] == 140400
    evidence = schedule["historical_ber0_evidence"]
    seen = set()
    winners = set()
    # The archive labels membership in the experiment 1 winner-centered grid.
    historical_multipliers = [Decimal("0.25"), Decimal("0.5"), 1, 2, 4]
    historical_grids = {
        key: [Decimal(str(model["strength"])) * factor for factor in historical_multipliers]
        for key, model in pristine_models.items()
    }
    in_grid_count = in_grid_quality_failures = 0
    with bound_config_path(evidence).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            condition, method = row["model_dataset"], row["method"]
            strength, update = Decimal(row["pilot_strength"]), int(row["updates"])
            key = condition, method, strength, update
            assert key not in seen, key
            seen.add(key)
            grid = historical_grids[condition, method]
            assert row["in_five_strength_grid"] == str(int(strength in grid)), key
            if strength in grid:
                in_grid_count += 1
                in_grid_quality_failures += row["quality_pass_any"] == "0"
            if method in design["embedding"]["direct_methods"]:
                cap = 0
            elif condition.endswith("_cifar100"):
                cap = class_cap
            else:
                cap = schedule["generation_updates"][-1]
            # Compare archived update counts to the budgets, using the original direct-insertion clock.
            assert 0 <= update <= cap, key
            if row["is_historical_winner"] == "1":
                assert strength == grid[2], key
                winners.add(key)
    expected_winners = set()
    for (condition, method), model in pristine_models.items():
        winner = read_bound_record(model["selection"])["winner"]
        assert winner["valid"] and winner["BER0"] and set(winner["recovery_statuses"]) == {"ok"}
        assert winner["candidate_id"] == model["selected_candidate_id"]
        strength = Decimal(str(winner["strength"]))
        assert strength == Decimal(str(model["strength"]))
        expected_winners.add((condition, method, strength, winner["updates"]))
    assert winners == expected_winners
    assert evidence["distinct_observations"] == len(seen) == 237
    assert evidence["observations_at_historical_strengths"] == in_grid_count == 82
    assert evidence["quality_failures_at_historical_strengths"] == in_grid_quality_failures == 18
    assert evidence["verified_historical_centers"] == len(winners) == 42


def verify_payload_and_evaluation(design):
    payload = design["payload"]
    message = payload["single_watermark"]
    registry = payload["physical_payloads_by_recipient"]
    assert len(registry) == len(set(registry.values())) == payload["recipient_count"] == 336
    assert payload["message_length_bits"] == 256
    logical_positions = payload["logical_to_physical"]
    physical_positions = payload["physical_to_logical"]
    assert sorted(physical_positions) == list(range(256))
    assert [logical_positions[i] for i in physical_positions] == list(range(256))
    trace_positions = payload["trace_physical_positions_in_logical_order"]
    assert trace_positions == logical_positions[:64]
    covered_pairs = set()
    digest = hashlib.sha256()
    for i in range(len(registry)):
        bits = registry[str(i)]
        assert bits.count("0") == bits.count("1") == 128 and len(bits) == 256, i
        block = [point for point, position in enumerate(trace_positions) if bits[position] == "0"]
        assert len(block) == 4, i
        covered_pairs.update(combinations(block, 2))
        digest.update(bytes(map(int, bits)))
    # 336 four-point blocks provide exactly 2016 pairs, the number of pairs on 64 points.
    assert covered_pairs == set(combinations(range(64), 2))
    assert digest.hexdigest() == payload["digests"]["physical_registry"]
    assert message["recipient_id"] == payload["codebook_validation_sampling"]["final_single_model_ids"][0] == 292
    assert registry["292"] == f'{int(message["hex_msb_first"], 16):0256b}'
    assert message["zeros"] == message["ones"] == 128
    assert set(message["targets"]) == set(design["method_display_names"])
    embedding = design["embedding"]
    assert set(embedding["direct_methods"]) == {"lerra", "tattooed"}
    assert set(embedding["trained_methods"]) == {"uchida", "greedy", "neuralmark", "deepmarks"}
    selection = design["strength"]["selection"]
    assert selection["required_bit_count"] == payload["message_length_bits"]
    assert selection["ranking"] == ["terminal_bit_errors_ascending", "strength_descending"]
    assert selection["quality_used_for_selection"] is False
    assert selection["no_valid_readout"] == "selection_failure"
    assert selection["max_selected_watermarked_models"] == design["strength"]["combination_count"]
    data = design["data"]
    verify_cifar_membership(data["source_CIFAR100"])
    assert data["target_FGVC_Aircraft"]["counts"] == {"train": 3334, "validation": 3333, "test": 3333}
    assert data["target_ArtBench"]["counts"] == {"train": 45000, "validation": 5000, "reserved_official_test": 10000}
    evaluation = design["generation_evaluation"]
    assert evaluation["pairs_per_split"] == 1000
    seeds = evaluation["seed_ranges_inclusive"]
    assert seeds["validation"][1] < seeds["test"][0]
    assert all(stop - start + 1 == 1000 for start, stop in (seeds["validation"], seeds["test"]))


def verify_robustness(design):
    robustness = design["robustness"]
    assert robustness["start_pair"] == ["W_T", "C_T"]
    assert robustness["branches_independent"] is True
    assert robustness["used_for_strength_selection"] is False
    branches = robustness["branches"]
    assert set(branches) == {"pruning", "quantization", "fine_tuning"}
    pruning, quantization = branches["pruning"], branches["quantization"]
    assert pruning["algorithm"] == "global_magnitude_stable_index" and pruning["fractions"] == [0.3, 0.7]
    assert quantization["algorithm"] == "symmetric_per_tensor_nearest_even" and quantization["bits"] == [8, 4]
    assert pruning["additional_optimizer_updates"] == quantization["additional_optimizer_updates"] == 0
    assert branches["fine_tuning"]["settings_ref"] == "#/fine_tuning"
    assert branches["fine_tuning"]["quality_ref"] == "#/post_ft_readout"
    assert design["fine_tuning"]["models"]["classification"]["dataset"] == "target_FGVC_Aircraft"
    assert robustness["readout"]["required_bit_count"] == design["payload"]["message_length_bits"]
    assert robustness["readout"]["payload_ref"] == "#/payload/single_watermark"
    scope = robustness["eligible_scope"]
    assert set(scope["applies_to"]) == {"pruning", "quantization"}
    assert scope["classifier_layers_pointer"] == "/config/modifications/nonlearning/0/layers"
    source_conditions = design["embedding"]["fixed_source_checkpoints"]
    classifier_lists = scope["classifier_layer_lists"]
    assert classifier_lists.keys() == {name for name in source_conditions if name.endswith("_cifar100")}
    for condition, binding in classifier_lists.items():
        archive = read_json(verify_file(PROJECT / binding["path_from_project_root"], binding))
        operations = archive["config"]["modifications"]["nonlearning"]
        layers = operations[0]["layers"]
        assert len(layers) == len(set(layers)) == binding["tensor_count"], condition
        assert all(operation["layers"] == layers for operation in operations), condition
        assert design["embedding"]["lessmark_classifier_backbones"]["conditions"][condition]["weight_name"] in layers
        assert {op["fraction"] for op in operations if op["algorithm"] == pruning["algorithm"]} == set(pruning["fractions"])
        assert {op["bits"] for op in operations if op["algorithm"] == quantization["algorithm"]} == set(quantization["bits"])

    # Each generator's archived unmodified layer scope is shared by the two operations.
    binding = scope["generation_reference_rows"]
    records = read_json(verify_file(PROJECT / binding["path_from_project_root"], binding))
    generators = design["embedding"]["native_generator_training"]["conditions"]
    checked = set()
    for row in records["rows"]:
        if row["method"] == "lerra" and row["modification"] == "prune30":
            condition = row["condition"]
            arch = generators[condition]["architecture"]
            saved_scope = read_bound_record(row["source_result"])["attack_scope"]
            physical = saved_scope["physical"]
            assert len(physical) == len(set(physical)) == scope["generation_tensor_counts"][arch], condition
            assert saved_scope["allowed_rows"] == scope["generator_output_rows"][arch], condition
            checked.add(condition)
    assert checked == generators.keys()


def main():
    if not __debug__:
        raise SystemExit("Run without -O to enable the validation checks.")
    design = read_json(CONFIG)
    for directory in ("code", "results"):
        relative = design["workspace"][f"{directory}_directory_relative_to_this_file"]
        destination = (CONFIG.parent / relative).resolve()
        assert destination == EXPERIMENT / directory and destination.is_dir(), destination
    source = read_json(bound_config_path(design["strength"]["historical_centers_source"]))
    models = {(row["condition"], row["method"]): row for row in source["rows"]}
    assert len(models) == len(source["rows"])
    grids = verify_strength_grid(design, models)
    verify_training_recipe(design)
    verify_historical_checkpoints(design, models)
    verify_payload_and_evaluation(design)
    verify_classifier_backbones(design)
    verify_robustness(design)
    affected = verify_native_training(design, grids)
    verify_photo_calibration(design["embedding"])
    print(json.dumps({"status": "design_verified", "clean_checkpoints": 7, "combinations": len(grids),
                      "strength_candidates": sum(map(len, grids.values())),
                      "lessmark_classifier_weights": [carrier["weight_name"] for carrier in
                                                     design["embedding"]["lessmark_classifier_backbones"]["conditions"].values()],
                      "historical_ber0_observations": 237, "recipient_id": 292,
                      "native_generator_combinations": affected,
                      "terminal_classification_epochs": design["checkpoint_schedule"]["classification_epochs"][0],
                      "terminal_generator_updates": design["checkpoint_schedule"]["generation_updates"][0],
                      "selection_order": design["strength"]["selection"]["ranking"],
                      "robustness_branches": list(design["robustness"]["branches"]),
                      "verified_source_photos": {"imagenet": 1000, "caltech256": 1000, "ffhq": 1000}}))


if __name__ == "__main__":
    main()
