import numpy as np
import json
import psutil
import time
import datetime
import torch
from pathlib import Path
from sacred import Experiment as Exp
from sacred.commands import print_config
from sacred.observers import FileStorageObserver

from paderbox.utils.random_utils import (
    LogTruncatedNormal, TruncatedExponential
)
from paderbox.transform.module_fbank import MelWarping
from paderbox.utils.nested import flatten, deflatten
from padertorch.train.hooks import LRAnnealingHook
from padertorch.train.trigger import AllTrigger, EndTrigger, NotTrigger
from padertorch.contrib.aw.optimizer import AdamW
from padertorch.train.trainer import Trainer
from padertorch import Configurable

from pb_sed.paths import storage_root, database_jsons_dir
from pb_sed.data_preparation.provider import DataProvider
from pb_sed.database.desed.provider import DESEDProvider
from pb_sed.database.audioset.provider import AudioSetProvider
from pb_sed.experiments.weak_label_transformer.tuning import ex as tuning

from padertorch.contrib.aw.predictor import PredictorHead
from padertorch.contrib.aw.transformer import TransformerEncoder
from padertorch.contrib.aw.transformer_blocks import RelativePositionalBiasFactory
from padertorch.contrib.aw.patch_embed import PatchEmbed
from padertorch.contrib.aw.trainer import AWTrainer

from padertorch.contrib.aw.transformer_blocks import AttentionBlockFactory

from padertorch.contrib.aw.positional_encoding import DisentangledPositionalEncoder
import lazy_dataset
from paderbox.utils.nested import flatten, deflatten

from pb_sed.models.weak_label.split_transformer import SplitTransformer

import paderbox as pb


ex_name = 'weak_label_transformer_training'
ex = Exp(ex_name)


@ex.config
def config():
    delay = 0
    debug = False
    dt = datetime.datetime.now()
    timestamp = dt.strftime('%Y-%m-%d-%H-%M-%S-{:02d}').format(
        int(dt.microsecond/10000)) + ('_debug' if debug else '')
    del dt
    group_name = timestamp
    database_name = 'desed'
    storage_dir = str(storage_root / 'weak_label_transformer' /
                      database_name / 'training' / group_name / timestamp)
    
    n_splits = 1

    resume = False
    if resume:
        assert Path(storage_dir).exists()
    else:
        # assert not Path(storage_dir).exists()
        print("assert disabled for storage_dir creation")
        # Path(storage_dir).mkdir(parents=True)

    init_ckpt_path = None
    freeze_norm_stats = True
    finetune_mode = init_ckpt_path is not None
    patch_size = [20, 1]
    patch_overlap = [0, 0]
    no_regularization = False
    debug_train_mode = 'single'
    use_lr_scheduler = False
    num_filters = 80

    # Data provider
    if database_name == 'desed':
        external_data = True
        batch_size = 4  # est. 15GB GPU memory
        data_provider = {
            'factory': DESEDProvider,
            'train_set': {
                'train_weak': 10 if external_data else 20,
                'train_strong': 10 if external_data else 0,
                'train_synthetic20': 2,
                'train_synthetic21': 1,
                'train_unlabel_in_domain': 0,
            },
            'cached_datasets': None if debug else ['train_weak', 'train_synthetic20'],
            'train_fetcher': {
                'batch_size': batch_size,
                'prefetch_workers': len(psutil.Process().cpu_affinity())-2,
                'min_dataset_examples_in_batch': {
                    'train_weak': int(3*batch_size/32),
                    'train_strong': int(6*batch_size/32) if external_data else 0,
                    'train_synthetic20': int(1*batch_size/32),
                    'train_synthetic21': int(2*batch_size/32),
                    'train_unlabel_in_domain': 0,
                },
            },
            'test_fetcher': {'batch_size': batch_size},
            'train_transform': {
                'provide_boundary_targets': True,
            },
            'storage_dir': storage_dir,
        }
        num_events = 10
        DESEDProvider.get_config(data_provider)

        validation_set_name = 'validation'
        validation_ground_truth_filepath = None
        eval_set_name = 'eval_public'
        eval_ground_truth_filepath = None

        num_iterations = int(
            40000 *
            (1+0.5*(data_provider['train_set']
                    ['train_unlabel_in_domain'] > 0)) * 16/batch_size
        )
        checkpoint_interval = int(2000 * 16/batch_size)
        summary_interval = 100
        lr = 1e-4
        n_back_off = 0
        back_off_patience = 10
        lr_decay_steps = [
            int(20000 * (1+0.5*(data_provider['train_set']
                ['train_unlabel_in_domain'] > 0)) * 16/batch_size)
        ] if n_back_off == 0 else []
        lr_decay_factor = 1/5
        lr_rampup_steps = None if finetune_mode else int(2000 * 16/batch_size)
        # TODO: find good value, is higher for transformer -> use deepnor or other method?
        gradient_clipping = 1 if finetune_mode else 1
        early_stopping_patience = None
    elif database_name == 'audioset':
        batch_size = 16  # est. 15GB GPU memory
        data_provider = {
            'factory': AudioSetProvider,
            'train_set': {
                'balanced_train': 1,
                'unbalanced_train': 1,
            },
            'train_fetcher': {
                'batch_size': batch_size,
                'prefetch_workers': len(psutil.Process().cpu_affinity())-2,
            },
            'min_class_examples_per_epoch': 0.01,
            'storage_dir': storage_dir,
        }
        num_events = 527
        AudioSetProvider.get_config(data_provider)

        validation_set_name = None
        validation_ground_truth_filepath = None
        eval_set_name = None
        eval_ground_truth_filepath = None

        num_iterations = int(1000000 * 16/batch_size)
        checkpoint_interval = int(10000 * 16/batch_size)
        summary_interval = int(1000 * 16/batch_size)
        lr = 1e-4  # 5e-5
        n_back_off = 0
        back_off_patience = 10
        lr_decay_steps = [
            int(600000 * 16/batch_size),
            int(800000 * 16/batch_size)
        ] if n_back_off == 0 else []
        lr_decay_factor = float(np.sqrt(.1))
        lr_rampup_steps = int(2000 * 16/batch_size)
        early_stopping_patience = None

        gradient_clipping = .1
    else:
        raise ValueError(f'Unknown database {database_name}.')
    filter_desed_test_clips = False
    hyper_params_tuning_batch_size = batch_size // 2

    # Trainer configuration
    
    max_grid_w = 600

    config_path = "/net/vol/werning/configs"

    model_config = pb.io.load(f'{config_path}/beats_{database_name}_model_cls_token.json')
    feature_extractor_config = pb.io.load(f'{config_path}/sed/desed/feature_extractor.json')['feature_extractor']
    model_config['strong_loss_weight'] = 0.0
    trainer = {
        'factory': AWTrainer,
        # 'clip_summary':  {
        #     'factory': 'padertorch.configurable.import_class',
        #     'name': 'pb_sed.experiments.weak_label_transformer.training.clip_summary'
        # },
        'model': {
            'factory': SplitTransformer,
            'feature_extractor': feature_extractor_config,
            'model': model_config,
            'splits': n_splits,
        },
        'optimizer': {
            'factory': AdamW,
            'lr': lr,
            'gradient_clipping': gradient_clipping,
            # 'weight_decay': 1e-6, # 0.05 TODO: set this value
        },
        'summary_trigger': (summary_interval, 'iteration'),
        'checkpoint_trigger': (checkpoint_interval, 'iteration'),
        'stop_trigger': (num_iterations, 'iteration'),
        'storage_dir': storage_dir,
    }

    # fine_tune parameters
    freeze_model = False

    Configurable.get_config(trainer)
    device = None
    track_emissions = False
    ex.observers.append(FileStorageObserver.create(trainer['storage_dir']))


def prepare(data_provider, trainer, filter_desed_test_clips):
    data_provider = DataProvider.from_config(data_provider)
    data_provider.train_transform.label_encoder.initialize_labels(
        dataset=data_provider.db.get_dataset(list(filter(
            lambda key: data_provider.train_set[key] > 0,
            data_provider.train_set.keys()
        ))),
        verbose=True
    )
    data_provider.test_transform.label_encoder.initialize_labels()
    trainer = Trainer.from_config(trainer)
    trainer.model.label_mapping = []
    for idx, label in sorted(data_provider.train_transform.label_encoder.inverse_label_mapping.items()):
        assert idx == len(trainer.model.label_mapping), (idx,
                                                         label, len(trainer.model.label_mapping))
        trainer.model.label_mapping.append(label.replace(', ', '__').replace(
            ' ', '').replace('(', '_').replace(')', '_').replace("'", ''))
    print('Params', sum(p.numel() for p in trainer.model.parameters()))

    if filter_desed_test_clips:
        with (database_jsons_dir / 'desed.json').open() as fid:
            desed_json = json.load(fid)
        filter_example_ids = {
            clip_id.rsplit('_', maxsplit=2)[0][1:]
            for clip_id in (
                list(desed_json['datasets']['validation'].keys())
                + list(desed_json['datasets']['eval_public'].keys())
            )
        }
    else:
        filter_example_ids = None
    train_set = data_provider.get_train_set(
        filter_example_ids=filter_example_ids)
    validate_set = data_provider.get_validate_set()

    return data_provider, trainer, train_set, validate_set


@ex.command
def debug_train(_run, debug, resume,
                data_provider, filter_desed_test_clips, trainer,
                n_back_off, back_off_patience, lr_decay_factor,
                early_stopping_patience, device, track_emissions, debug_train_mode
                ):
    print()
    print('##### Debug Training #####')
    print()
    print_config(_run)

    data_provider, trainer, train_set, validate_set = prepare(
        data_provider, trainer, filter_desed_test_clips)

    if validate_set is not None:
        # trainer.test_run(train_set, validate_set)
        trainer.register_validation_hook(
            validate_set, metric='macro_fscore_weak', maximize=True,
            back_off_patience=back_off_patience,
            n_back_off=n_back_off,
            lr_update_factor=lr_decay_factor,
            early_stopping_patience=early_stopping_patience,
        )

    if debug_train_mode == 'single':
        example = next(iter(train_set))
        train_set = lazy_dataset.from_list([example]).tile(1000)
    elif debug_train_mode == 'independent':
        example = next(iter(train_set))
        import numpy as np

        def remove_data(example):
            shape = example['stft'].shape
            example['stft'] = np.random.normal(
                size=shape, scale=5.6).astype(np.float32)
            return example
        train_set = train_set.map(remove_data)

    trainer.train(
        train_set, resume=resume, device=device,
        track_emissions=track_emissions,
    )


@ex.command
def tune(
        _run, debug, resume, delay,
        data_provider, filter_desed_test_clips, trainer, lr_rampup_steps,
        n_back_off, back_off_patience, lr_decay_steps, lr_decay_factor,
        early_stopping_patience,
        init_ckpt_path, freeze_norm_stats,
        validation_set_name, validation_ground_truth_filepath,
        eval_set_name, eval_ground_truth_filepath,
        device, track_emissions, hyper_params_tuning_batch_size,
        use_lr_scheduler
):
    if validation_set_name is not None:
        tuning.run(
            config_updates={
                'debug': debug,
                'model_dirs': [str(trainer['storage_dir'])],
                'validation_set_name': validation_set_name,
                'validation_ground_truth_filepath': validation_ground_truth_filepath,
                'eval_set_name': eval_set_name,
                'eval_ground_truth_filepath': eval_ground_truth_filepath,
                'data_provider': {
                    'test_fetcher': {
                        'batch_size': hyper_params_tuning_batch_size,
                    }
                },
            }
        )

@ex.command
def fine_tune(
        _run, debug, resume, delay,
        data_provider, filter_desed_test_clips, trainer, lr_rampup_steps,
        n_back_off, back_off_patience, lr_decay_steps, lr_decay_factor,
        early_stopping_patience,
        init_ckpt_path, freeze_norm_stats,
        validation_set_name, validation_ground_truth_filepath,
        eval_set_name, eval_ground_truth_filepath,
        device, track_emissions, hyper_params_tuning_batch_size,
        use_lr_scheduler
):
    print()
    print('##### Training #####')
    print()
    print_config(_run)
    assert (n_back_off == 0) or (len(lr_decay_steps)
                                 == 0), (n_back_off, lr_decay_steps)
    if delay > 0:
        print(f'Sleep for {delay} seconds.')
        time.sleep(delay)

    data_provider, trainer, train_set, validate_set = prepare(
        data_provider, trainer, filter_desed_test_clips)

    print('Params', sum(p.numel() for p in trainer.model.parameters()))

    assert init_ckpt_path is not None
    print('Load init params')
    state_dict = torch.load(init_ckpt_path, map_location='cpu')[
        'model']
    for key in list(state_dict.keys()):
        if key.startswith('model.predictor'):
            state_dict.pop(key)  # model.model.predictor
    trainer.model.load_state_dict(state_dict, strict=False)

    if validate_set is not None:
        trainer.test_run(train_set, validate_set, device=device)
        trainer.register_validation_hook(
            validate_set, metric='macro_fscore_weak', maximize=True,
            back_off_patience=back_off_patience,
            n_back_off=n_back_off,
            lr_update_factor=lr_decay_factor,
            early_stopping_patience=early_stopping_patience,
        )

    breakpoints = []
    if lr_rampup_steps is not None:
        breakpoints += [(0, 0.), (lr_rampup_steps, 1.)]
    for i, lr_decay_step in enumerate(lr_decay_steps):
        breakpoints += [(lr_decay_step, lr_decay_factor**i),
                        (lr_decay_step, lr_decay_factor**(i+1))]
    if len(breakpoints) > 0:
        if isinstance(trainer.optimizer, dict):
            names = sorted(trainer.optimizer.keys())
        else:
            names = [None]
        for name in names:
            trainer.register_hook(LRAnnealingHook(
                trigger=AllTrigger(
                    (100, 'iteration'),
                    NotTrigger(EndTrigger(
                        breakpoints[-1][0]+100, 'iteration')),
                ),
                breakpoints=breakpoints,
                unit='iteration',
                name=name,
            ))
    trainer.train(
        train_set, resume=resume, device=device,
        track_emissions=track_emissions,
    )

    if validation_set_name is not None:
        tuning.run(
            config_updates={
                'debug': debug,
                'model_dirs': [str(trainer.storage_dir)],
                'validation_set_name': validation_set_name,
                'validation_ground_truth_filepath': validation_ground_truth_filepath,
                'eval_set_name': eval_set_name,
                'eval_ground_truth_filepath': eval_ground_truth_filepath,
                'data_provider': {
                    'test_fetcher': {
                        'batch_size': hyper_params_tuning_batch_size,
                    }
                },
            }
        )

@ex.automain
def train(
        _run, debug, resume, delay,
        data_provider, filter_desed_test_clips, trainer, lr_rampup_steps,
        n_back_off, back_off_patience, lr_decay_steps, lr_decay_factor,
        early_stopping_patience,
        init_ckpt_path, freeze_norm_stats,
        validation_set_name, validation_ground_truth_filepath,
        eval_set_name, eval_ground_truth_filepath,
        device, track_emissions, hyper_params_tuning_batch_size,
        use_lr_scheduler
):
    print()
    print('##### Training #####')
    print()
    print_config(_run)
    assert (n_back_off == 0) or (len(lr_decay_steps)
                                 == 0), (n_back_off, lr_decay_steps)
    if delay > 0:
        print(f'Sleep for {delay} seconds.')
        time.sleep(delay)

    data_provider, trainer, train_set, validate_set = prepare(
        data_provider, trainer, filter_desed_test_clips)

    print('Params', sum(p.numel() for p in trainer.model.parameters()))

    if init_ckpt_path is not None:
        print('Load init params')
        state_dict = torch.load(init_ckpt_path, map_location='cpu')[
            'model']
        trainer.model.load_state_dict(state_dict, strict=False)

    if validate_set is not None:
        trainer.test_run(train_set, validate_set, device=device)
        trainer.register_validation_hook(
            validate_set, metric='macro_fscore_weak', maximize=True,
            back_off_patience=back_off_patience,
            n_back_off=n_back_off,
            lr_update_factor=lr_decay_factor,
            early_stopping_patience=early_stopping_patience,
        )

    breakpoints = []
    if lr_rampup_steps is not None:
        breakpoints += [(0, 0.), (lr_rampup_steps, 1.)]
    for i, lr_decay_step in enumerate(lr_decay_steps):
        breakpoints += [(lr_decay_step, lr_decay_factor**i),
                        (lr_decay_step, lr_decay_factor**(i+1))]
    if len(breakpoints) > 0:
        if isinstance(trainer.optimizer, dict):
            names = sorted(trainer.optimizer.keys())
        else:
            names = [None]
        for name in names:
            trainer.register_hook(LRAnnealingHook(
                trigger=AllTrigger(
                    (100, 'iteration'),
                    NotTrigger(EndTrigger(
                        breakpoints[-1][0]+100, 'iteration')),
                ),
                breakpoints=breakpoints,
                unit='iteration',
                name=name,
            ))
    trainer.train(
        train_set, resume=resume, device=device,
        track_emissions=track_emissions,
    )

    if validation_set_name is not None:
        tuning.run(
            config_updates={
                'debug': debug,
                'model_dirs': [str(trainer.storage_dir)],
                'validation_set_name': validation_set_name,
                'validation_ground_truth_filepath': validation_ground_truth_filepath,
                'eval_set_name': eval_set_name,
                'eval_ground_truth_filepath': eval_ground_truth_filepath,
                'data_provider': {
                    'test_fetcher': {
                        'batch_size': hyper_params_tuning_batch_size,
                    }
                },
            }
        )


