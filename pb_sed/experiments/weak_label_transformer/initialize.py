# initialize model for later training
import numpy as np
import time
import datetime
import torch
from pathlib import Path
from sacred import Experiment as Exp
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from pathlib import Path

from paderbox.utils.random_utils import (
    LogTruncatedNormal, TruncatedExponential
)
from paderbox.transform.module_fbank import MelWarping
from padertorch.contrib.aw.optimizer import AdamW
from padertorch.train.trainer import Trainer
from padertorch import Configurable

from pb_sed.models.weak_label.transformer import Transformer
from pb_sed.paths import storage_root, database_jsons_dir
from pb_sed.experiments.weak_label_transformer.tuning import ex as tuning

from padertorch.contrib.aw.predictor import PredictorHead
from padertorch.contrib.aw.transformer import TransformerEncoder
from padertorch.contrib.aw.transformer_blocks import RelativePositionalBiasFactory
from padertorch.contrib.aw.patch_embed import PatchEmbed
from padertorch.contrib.aw.positional_encoding import ConvolutionalPositionalEncoder

from padertorch.contrib.aw.transformer_blocks import AttentionBlockFactory

from padertorch.contrib.aw.positional_encoding import DisentangledPositionalEncoder
from paderbox.io.new_subdir import NameGenerator
from padertorch.contrib.aw.name_generator import animal_names, food_names, thing_names
import sacred
from pb_sed.experiments.weak_label_transformer.training import prepare

sacred.SETTINGS.CONFIG.READ_ONLY_CONFIG = False
ex_name = 'weak_label_transformer_training'
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
    database_name = 'desed'
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

    # Trainer configuration
    max_grid_w = 600
    sample_rate = 16000
    stft_size = 1024
    num_events = 10
    strong_fwd_bwd_loss_weight = 0
    
    deep_norm = False
    use_relative_positional_bias = True
    attention_block_implementation = "own"
    qkv_bias = True
    patch_embed_dim = 512
    net_config = 'ViT-base'
    if net_config == 'ViT-base':
        # 86M parameters
        embed_dim = 768
        depth = 12
        num_heads = 12
    elif net_config == 'ViT-small':
        # 22M parameters
        embed_dim = 384
        depth = 12
        num_heads = 6
    elif net_config == 'DeiT-base':
        # 86M parameters
        embed_dim = 768
        depth = 12
        num_heads = 12
        use_relative_positional_bias = False
        attention_block_implementation = "torch"
        qkv_bias = False
        patch_embed_dim = embed_dim
        # pos_enc = {'factory': Learned2DPositionalEncoding}
    else:
        raise ValueError(f'Unknown net config {net_config}.')

    # pos_enc = {
    #     'factory': DisentangledPositionalEncoder,
    #     'grid': [5, max_grid_w // patch_size[1]],
    #     'use_class_token': False,
    #     'embed_dim': embed_dim,
    # }
    # 'pos_enc = {
    #     'factory': SinCos1DPositionalEncoder,
    #     'sequence_length': 5 * max_grid_w,  # num_mel_filters // patch_size[0] # maximum number of patches
    #     'use_class_token': False,
    #     'embed_dim': embed_dim,
    # }
    pos_enc = {
        'factory': ConvolutionalPositionalEncoder,
        'kernel_size': 128,
        'groups': 16,
        'dropout': 0.1,  # used to initialize the conv weight
        'use_class_token': False,
        'embed_dim': embed_dim,
    }
    # pos_enc = {
    #     'factory': DummyPositionalEncoder,
    #     'embed_dim': embed_dim,
    #     'use_class_token': False,
    # }
    # '/net/home/werning/pretrained/mae_pretrain_vit_base.pth'
    patch_embed_init_path = None
    init_ckpt_path = None
    pretrained_weights = None
    if pretrained_weights == 'DeiT-base-ImageNet':
        init_ckpt_path = 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-cd65a155.pth'
    elif pretrained_weights == 'PaSST-base':
        # 128 mel
        num_filters = 128
        init_ckpt_path = 'passt-s-f128-p16-s16-ap.468.pt'
        # no relative attention bias
        assert net_config == 'DeiT-base'
    elif pretrained_weights == 'BEATs-base_iter3_plus':
        init_ckpt_path = 'BEATs/BEATs_iter3.pt'  # 'BEATs/BEATs_iter3_plus_AS2M.pt'
        # pos enc: 
        # relative attention bias
        # convolutional positional encoding
    elif pretrained_weights == 'BEATs-base_iter3_plus_AS2M':
        init_ckpt_path = 'BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt'
    if pretrained_weights is not None:
        import os
        assert 'PRETRAINED_MODELS_DIR' in os.environ
        pretrained_dir = os.environ['PRETRAINED_MODELS_DIR']
        init_ckpt_path = os.path.join(pretrained_dir, init_ckpt_path)
        del os
        patch_embed_init_path = init_ckpt_path

    # TODO: fix
    if True or pretrained_weights == 'PaSST-base':
        # learned positional encoding, disentangled over frequency and time
        pos_enc =  {
            'factory': DisentangledPositionalEncoder,
            'grid': [num_filters// patch_size[0], max_grid_w // patch_size[1]],
            'use_class_token': False,
            'embed_dim': embed_dim,
            'init': init_ckpt_path,
            'h_enc': 'learned',
            'w_enc': 'learned',
        }

    trainer = {
        'factory': Trainer,
        'model': {
            'factory': Transformer,
            'feature_extractor': {
                'sample_rate':
                    sample_rate,
                'stft_size': stft_size,
                'number_of_filters': num_filters,
                'frequency_warping_fn': {
                    'factory': MelWarping,
                    'warp_factor_sampling_fn': {
                        'factory': LogTruncatedNormal,
                        'scale': .08,
                        'truncation': np.log(1.3),
                    },
                    'boundary_frequency_ratio_sampling_fn': {
                        'factory': TruncatedExponential,
                        'scale': .5,
                        'truncation': 5.,
                    },
                    'highest_frequency': sample_rate/2,
                },
                # 'blur_sigma': .5,
                'n_time_masks': 1,
                'max_masked_time_steps': 70 if not no_regularization else 0,
                'max_masked_time_rate': .2 if not no_regularization else 0,
                'n_frequency_masks': 1 if not no_regularization else 0,
                'max_masked_frequency_bands': 20 if not no_regularization else 0,
                'max_masked_frequency_rate': .2 if not no_regularization else 0,
                'max_noise_scale': .2 if not no_regularization else 0,
            },
            'encoder': {
                'factory': TransformerEncoder,
                'embed_dim': embed_dim,
                'depth': depth,
                'num_heads': num_heads,
                'dropout': 0.1 if not no_regularization else 0,
                'attn_dropout': 0.1 if not no_regularization else 0,
                'layer_dropout': 0.0 if not no_regularization else 0,
                'forward': True,
                'backward': False,
                'block_factory': {
                    'factory': AttentionBlockFactory,
                    'implementation': attention_block_implementation,
                    'style': 'post-ln' if deep_norm else 'pre-ln',
                    'qkv_bias': qkv_bias,
                },
                'init_mode': 'deep_norm' if deep_norm else 'xlm',
                'rel_pos_bias_factory': {
                    'factory': RelativePositionalBiasFactory,
                    'style': '2d',
                    'grid': [5, max_grid_w // patch_size[1]],
                    # 'gated': True,
                    # 'num_buckets': 320,
                    # 'max_distance': 800,
                    # 'gate_dim': 8,
                } if use_relative_positional_bias else False,
            },
            'encoder_bwd': {
                'factory': TransformerEncoder,
                'embed_dim': embed_dim,
                'depth': depth,
                'num_heads': num_heads,
                'dropout': 0.1 if not no_regularization else 0,
                'attn_dropout': 0.1 if not no_regularization else 0,
                'layer_dropout': 0.0 if not no_regularization else 0,
                'forward': False,
                'backward': True,
                'block_factory': {
                    'factory': AttentionBlockFactory,
                    'implementation': attention_block_implementation,
                    'style': 'post-ln' if deep_norm else 'pre-ln',
                    'qkv_bias': qkv_bias,
                },
                'init_mode': 'deep_norm' if deep_norm else 'xlm',
                'rel_pos_bias_factory': {
                    'factory': RelativePositionalBiasFactory,
                    'style': '2d',
                    'grid': [5, max_grid_w // patch_size[1]],
                    # 'gated': True,
                    # 'num_buckets': 320,
                    # 'max_distance': 800,
                    # 'gate_dim': 8,
                } if use_relative_positional_bias else False,
            },
            'pos_enc': pos_enc,
            'patch_embed': {'factory': PatchEmbed,
                            'patch_size': patch_size,
                            'patch_overlap': patch_overlap,
                            'embed_dim': patch_embed_dim,
                            'output_dim': embed_dim if patch_embed_dim != embed_dim else None,
                            # should be necessary for variable-length ViT, independent of grid width
                            'flatten_transpose': False,  # True,
                            'bias': False,
                            # 'learnable': False,
                            'init_path': patch_embed_init_path,
                            },
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
                              },
            'share_weights_transformer': False,
            'share_weights_classifier': False,
            'labelwise_metrics': ('fscore_weak',),
            'strong_fwd_bwd_loss_weight': strong_fwd_bwd_loss_weight,
            'init_path': init_ckpt_path,
        },
        'optimizer': {
            'factory': AdamW,
        },
        'summary_trigger': (1, 'iteration'),
        'checkpoint_trigger': (1, 'iteration'),
        'stop_trigger': (1, 'iteration'),
        'storage_dir': storage_dir,
    }

    # fine_tune parameters
    freeze_model = False

    Configurable.get_config(trainer)
    device = None
    track_emissions = False
    ex.observers.append(FileStorageObserver.create(trainer['storage_dir']))


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
        trainer.test_run(train_set, validate_set)
        trainer.register_validation_hook(
            validate_set, metric='macro_fscore_weak', maximize=True,
            back_off_patience=back_off_patience,
            n_back_off=n_back_off,
            lr_update_factor=lr_decay_factor,
            early_stopping_patience=early_stopping_patience,
        )
    # TODO: manually create best model checkpoint symlink
    trainer.train(
        train_set, resume=resume, device=device,
        track_emissions=track_emissions,
    )

    if validation_set_name is not None:
        tuning.run(
            config_updates={
                ''
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
