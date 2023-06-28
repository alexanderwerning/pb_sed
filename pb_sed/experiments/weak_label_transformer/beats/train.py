import numpy as np
import json
import psutil
import time
import datetime
import torch
from pathlib import Path
from pathlib import Path

from paderbox.utils.random_utils import LogTruncatedNormal, TruncatedExponential
from paderbox.transform.module_fbank import MelWarping
from padertorch.train.hooks import LRAnnealingHook
from padertorch.train.trigger import AllTrigger, EndTrigger, NotTrigger
from padertorch.contrib.aw.optimizer import AdamW
from padertorch.train.trainer import Trainer
from padertorch import Configurable

from pb_sed.paths import storage_root, database_jsons_dir
from pb_sed.database.desed.provider import DESEDProvider
from pb_sed.experiments.weak_label_transformer.tuning import ex as tuning

from padertorch.contrib.je.modules.features import NormalizedLogMelExtractor

from paderbox.io.new_subdir import NameGenerator

from pb_sed.experiments.weak_label_transformer.beats.model import SEDModel

from padertorch.contrib.aw.name_generator import animal_names, food_names, thing_names

from padertorch.contrib.aw.transformer_models import BEATsModel
from torch import nn
from einops import rearrange

from pb_sed.experiments.weak_label_transformer.beats.model import ViTPredictor

def main():
    debug = False
    print(f"Debug {debug}")
    # todo: set name for fine_tuning based on old ensemble name
    group_name = NameGenerator(lists=(animal_names, food_names, thing_names))()
    dt = datetime.datetime.now()
    timestamp = dt.strftime("%Y-%m-%d-%H-%M-%S-{:02d}").format(
        int(dt.microsecond / 10000)
    ) + ("_debug" if debug else "")

    database_name = "desed"
    storage_dir = str(
        storage_root
        / "weak_label_transformer"
        / database_name
        / "training"
        / group_name
        / timestamp
    )
    resume = False
    if resume:
        assert Path(storage_dir).exists()
    else:
        assert not Path(storage_dir).exists()
        Path(storage_dir).mkdir(parents=True)

    init_ckpt_path = None
    finetune_mode = init_ckpt_path is not None
    num_filters = 80
    max_grid_w = 600

    external_data = True
    batch_size = 32  # est. 15GB GPU memory
    data_provider_config = {
        "factory": DESEDProvider,
        "train_set": {
            "train_weak": 10 if external_data else 20,
            "train_strong": 10 if external_data else 0,
            "train_synthetic20": 2,
            "train_synthetic21": 1,
            "train_unlabel_in_domain": 0,
        },
        "cached_datasets": None if debug else ["train_weak", "train_synthetic20"],
        "train_fetcher": {
            "batch_size": batch_size,
            "prefetch_workers": len(psutil.Process().cpu_affinity()) - 2,
            "min_dataset_examples_in_batch": {
                "train_weak": int(3 * batch_size / 32),
                "train_strong": int(6 * batch_size / 32) if external_data else 0,
                "train_synthetic20": int(1 * batch_size / 32),
                "train_synthetic21": int(2 * batch_size / 32),
                "train_unlabel_in_domain": 0,
            },
        },
        "train_transform": {
            "provide_boundary_targets": True,
        },
        "storage_dir": storage_dir,
    }
    DESEDProvider.get_config(data_provider_config)
    data_provider = Configurable.from_config(data_provider_config)
    num_events = 10

    validation_set_name = "validation"
    validation_ground_truth_filepath = None
    eval_set_name = "eval_public"
    eval_ground_truth_filepath = None

    num_iterations = int(
        40000
        * (1 + 0.5 * (data_provider_config["train_set"]["train_unlabel_in_domain"] > 0))
        * 16
        / batch_size
    )
    checkpoint_interval = int(2000 * 16 / batch_size)
    summary_interval = 100
    lr = 1e-4
    n_back_off = 0
    back_off_patience = 10
    lr_decay_steps = (
        [
            int(
                20000
                * (
                    1
                    + 0.5 * (data_provider_config["train_set"]["train_unlabel_in_domain"] > 0)
                )
                * 16
                / batch_size
            )
        ]
        if n_back_off == 0
        else []
    )
    lr_decay_factor = 1 / 5
    lr_rampup_steps = None if finetune_mode else int(2000 * 16 / batch_size)

    gradient_clipping = 1 if finetune_mode else 1
    strong_loss_weight = 1.0
    early_stopping_patience = None

    hyper_params_tuning_batch_size = batch_size // 2
    

    desed_num_classes = 10
    beats_embed_dim = 768
    predictor = ViTPredictor(grid_h=num_filters//16,
                             embed_dim=beats_embed_dim,
                             num_classes=desed_num_classes)

    encoder = BEATsModel(pretrained_dir='/net/vol/werning/pretrained/BEATs/BEATs_iter3.pt',
                         load_config_from_checkpoint=True, config={'use_class_token': True})

    feature_extractor = Configurable.from_config(
        {
            "factory": NormalizedLogMelExtractor,
            "sample_rate": 16_000,
            "stft_size": data_provider.train_transform.stft.size,
            "number_of_filters": num_filters,
            "frequency_warping_fn": {
                "factory": MelWarping,
                "warp_factor_sampling_fn": {
                    "factory": LogTruncatedNormal,
                    "scale": 0.08,
                    "truncation": np.log(1.3),
                },
                "boundary_frequency_ratio_sampling_fn": {
                    "factory": TruncatedExponential,
                    "scale": 0.5,
                    "truncation": 5.0,
                },
                "highest_frequency": data_provider_config["audio_reader"]["target_sample_rate"]
                / 2,
            },
            "n_time_masks": 1,
            "max_masked_time_steps": 70,
            "max_masked_time_rate": 0.2,
            "n_frequency_masks": 1,
            "max_masked_frequency_bands": 20,
            "max_masked_frequency_rate": 0.2,
            "max_noise_scale": 0.2,
        }
    )
    model = SEDModel(feature_extractor,
                     encoder=encoder,
                     predictor=predictor,
                     strong_loss_weight=strong_loss_weight,)
    # model_config = {'factory':SEDModel, 'feature_extractor': {
    #         "factory": NormalizedLogMelExtractor,
    #         "sample_rate": 16_000,
    #         "stft_size": data_provider.train_transform.stft.size,
    #         "number_of_filters": num_filters,
    #         "frequency_warping_fn": {
    #             "factory": MelWarping,
    #             "warp_factor_sampling_fn": {
    #                 "factory": LogTruncatedNormal,
    #                 "scale": 0.08,
    #                 "truncation": np.log(1.3),
    #             },
    #             "boundary_frequency_ratio_sampling_fn": {
    #                 "factory": TruncatedExponential,
    #                 "scale": 0.5,
    #                 "truncation": 5.0,
    #             },
    #             "highest_frequency": data_provider_config["audio_reader"]["target_sample_rate"]
    #             / 2,
    #         },
    #         "n_time_masks": 1,
    #         "max_masked_time_steps": 70,
    #         "max_masked_time_rate": 0.2,
    #         "n_frequency_masks": 1,
    #         "max_masked_frequency_bands": 20,
    #         "max_masked_frequency_rate": 0.2,
    #         "max_noise_scale": 0.2,
    #     },
    #     'encoder':{'factory': 'BEATsModel',
    #              'pretrained_dir': '/net/vol/werning/pretrained/BEATs/BEATs_iter3.pt',
    #              'load_config_from_checkpoint': True},
    #     'predictor': {'factory': ViTPredictor,
    #                 'grid_h': num_filters//16,
    #                 'embed_dim': beats_embed_dim,
    #                 'num_classes': desed_num_classes}
    #     }
    # import paderbox
    # paderbox.io.dump(Configurable.get_config(model_config), "/net/vol/werning/configs/model_config.json")

    optimizer = AdamW(lr=lr, gradient_clipping=gradient_clipping)
            # 'weight_decay': 1e-6, # 0.05 TODO: set this value

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        summary_trigger=(summary_interval, "iteration"),
        checkpoint_trigger=(checkpoint_interval, "iteration"),
        stop_trigger=(num_iterations, "iteration"),
        storage_dir=storage_dir,
    )

    device = None

    data_provider.train_transform.label_encoder.initialize_labels(
        dataset=data_provider.db.get_dataset(
            list(
                filter(
                    lambda key: data_provider.train_set[key] > 0,
                    data_provider.train_set.keys(),
                )
            )
        ),
        verbose=True,
    )
    data_provider.test_transform.label_encoder.initialize_labels()
    trainer.model.label_mapping = []
    for idx, label in sorted(
        data_provider.train_transform.label_encoder.inverse_label_mapping.items()
    ):
        assert idx == len(trainer.model.label_mapping), (
            idx,
            label,
            len(trainer.model.label_mapping),
        )
        trainer.model.label_mapping.append(
            label.replace(", ", "__")
            .replace(" ", "")
            .replace("(", "_")
            .replace(")", "_")
            .replace("'", "")
        )
    print("Params", sum(p.numel() for p in trainer.model.parameters()))

    train_set = data_provider.get_train_set(filter_example_ids=None)
    validate_set = data_provider.get_validate_set()

    print()
    print("##### Training #####")
    print()
    assert (n_back_off == 0) or (len(lr_decay_steps) == 0), (n_back_off, lr_decay_steps)

    print("Params", sum(p.numel() for p in trainer.model.parameters()))

    if init_ckpt_path is not None:
        print("Load init params")
        state_dict = torch.load(init_ckpt_path, map_location="cpu")["model"]
        trainer.model.load_state_dict(state_dict, strict=False)

    if validate_set is not None:
        trainer.test_run(train_set, validate_set)
        trainer.register_validation_hook(
            validate_set,
            metric="macro_fscore_weak",
            maximize=True,
            back_off_patience=back_off_patience,
            n_back_off=n_back_off,
            lr_update_factor=lr_decay_factor,
            early_stopping_patience=early_stopping_patience,
        )
    breakpoints = []
    if lr_rampup_steps is not None:
        breakpoints += [(0, 0.0), (lr_rampup_steps, 1.0)]
    for i, lr_decay_step in enumerate(lr_decay_steps):
        breakpoints += [
            (lr_decay_step, lr_decay_factor**i),
            (lr_decay_step, lr_decay_factor ** (i + 1)),
        ]
    if len(breakpoints) > 0:
        if isinstance(trainer.optimizer, dict):
            names = sorted(trainer.optimizer.keys())
        else:
            names = [None]
        for name in names:
            trainer.register_hook(
                LRAnnealingHook(
                    trigger=AllTrigger(
                        (100, "iteration"),
                        NotTrigger(EndTrigger(breakpoints[-1][0] + 100, "iteration")),
                    ),
                    breakpoints=breakpoints,
                    unit="iteration",
                    name=name,
                )
            )
    trainer.train(
        train_set,
        resume=resume,
        device=device
    )

if __name__ == "__main__":
    main()