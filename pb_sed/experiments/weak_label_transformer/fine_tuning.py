
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
from pathlib import Path

from paderbox.transform.module_fbank import MelWarping
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
from padertorch.contrib.aw.segmenter import TimeDomainViTSegmenter
from padertorch.contrib.je.data.transforms import STFT

from padertorch.contrib.aw.trainer import AWTrainer

import lazy_dataset
from paderbox.utils.nested import flatten, deflatten
from paderbox.io.new_subdir import NameGenerator
from padertorch.contrib.aw.name_generator import animal_names, food_names, thing_names
import paderbox
import sacred
from pb_sed.experiments.weak_label_transformer.training import prepare

sacred.SETTINGS.CONFIG.READ_ONLY_CONFIG = False
ex_name = 'weak_label_transformer_fine_tuning'
ex = Exp(ex_name)


@ex.config
def config():
    delay = 0
    debug = False
    # todo: set name for fine_tuning based on old ensemble name
    group_name = NameGenerator(lists=(animal_names, food_names, thing_names))()
    dt = datetime.datetime.now()
    timestamp = dt.strftime('%Y-%m-%d-%H-%M-%S-{:02d}').format(
        int(dt.microsecond/10000)) + ('_debug' if debug else '')
    del dt
    # group_name = timestamp
    database_name = 'audioset'
    storage_dir = str(storage_root / 'weak_label_transformer' /
                      database_name / 'training' / group_name / timestamp)
    resume = False
    if resume:
        assert Path(storage_dir).exists()
    else:
        assert not Path(storage_dir).exists()
        Path(storage_dir).mkdir(parents=True)

    init_ckpt_path = None
    freeze_norm_stats = True
    finetune_mode = init_ckpt_path is not None
    patch_size = [16, 16]
    patch_overlap = [0, 0]
    no_regularization = False
    debug_train_mode = 'single'
    use_lr_scheduler = False
    num_filters = 80

    segmenter = {
        'factory': TimeDomainViTSegmenter,
        'pad_last': False,
        'patch_size': patch_size,
        'patch_overlap': patch_overlap,
        'max_grid_w': 600,  # 10s audio @16kHz with 320 samples shift + 100
        'allow_shorter_segments': True,
        'stft': {
            'factory': STFT,
            'shift': 320,
            'window_length': 960,
            'size': 1024,
            'fading': 'half',
            'pad': True,
            'alignment_keys': ['events'],
        },
    }

    # Data provider
    if database_name == 'desed':
        external_data = True
        batch_size = 32  # est. 15GB GPU memory
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
            # 'train_segmenter': segmenter,
            # 'test_segmenter': segmenter,
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
        strong_fwd_bwd_loss_weight = 1.
        early_stopping_patience = None
    elif database_name == 'audioset':
        batch_size = 32  # est. 15GB GPU memory
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
            'train_segmenter': segmenter,
            'test_segmenter': segmenter,
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
        strong_fwd_bwd_loss_weight = 0.
    else:
        raise ValueError(f'Unknown database {database_name}.')
    filter_desed_test_clips = False
    hyper_params_tuning_batch_size = batch_size // 2

    # Trainer configuration
    trainer = {
        'factory': AWTrainer,
        'clip_summary':  {
            'factory': 'padertorch.configurable.import_class',
            'name': 'pb_sed.experiments.weak_label_transformer.training.clip_summary'
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
    config_path = Path(init_ckpt_path).parent.parent/ "1/config.json"
    config = paderbox.io.load(config_path)
    predictors = trainer['model']['predictor'], trainer['model']['predictor_bwd']
    trainer['model'] = config['trainer']['model']
    embed_dim = config['trainer']['model']['patch_embed']['embed_dim'] or config['trainer']['model']['patch_embed']['output_dim']
    new_predictors = True  # add new predictor heads for the new dataset
    if new_predictors:
        predictors = {
            'predictor': {'factory': PredictorHead,
                          'patch_embed_dim': embed_dim,
                          'num_classes': num_events,
                          'classifier_hidden_dims': [],
                          'pooling_op': 'mean',
                          'pooling_num_patches': 5,
                          'apply_softmax': False,
                          },
            'predictor_bwd': {'factory': PredictorHead,
                              'patch_embed_dim': embed_dim,
                              'num_classes': num_events,
                              'classifier_hidden_dims': [],
                              'pooling_op': 'mean',
                              'pooling_num_patches': 5,
                              'apply_softmax': False,
                              }
        }
        trainer['model'].update(predictors)
    freeze_model = False

    Configurable.get_config(trainer)
    device = None
    track_emissions = False
    ex.observers.append(FileStorageObserver.create(trainer['storage_dir']))


@ex.automain
def fine_tune(_run, debug, resume, delay,
              data_provider, filter_desed_test_clips, trainer, lr_rampup_steps,
              n_back_off, back_off_patience, lr_decay_steps, lr_decay_factor,
              early_stopping_patience,
              init_ckpt_path, freeze_norm_stats,
              validation_set_name, validation_ground_truth_filepath,
              eval_set_name, eval_ground_truth_filepath,
              device, track_emissions, hyper_params_tuning_batch_size,
              use_lr_scheduler, freeze_model):
    # load weights without classifier, feature_extractor
    assert init_ckpt_path is not None
    print()
    print('##### Fine-Tuning #####')
    print()
    
    data_provider, trainer, train_set, validate_set = prepare(
        data_provider, trainer, filter_desed_test_clips)
    print(f'Load init params from {init_ckpt_path}')
    state_dict = deflatten(torch.load(
        init_ckpt_path, map_location='cpu')['model'])
    # load forward transformer, backward transformer, patch_embedding, and position_embedding
    trainer.model.load_encoder_state_dict(state_dict)

    if freeze_model:
        trainer.model.encoder.requires_grad_(False)
        trainer.model.encoder_bwd.requires_grad_(False)
        trainer.model.patch_embed.requires_grad_(False)
        trainer.model.pos_enc.requires_grad_(False)

    # start training with new classifier
    if validate_set is not None:
        trainer.test_run(train_set, validate_set)
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
