from functools import partial
import numpy as np
import torch
from torch import nn
from einops import rearrange
from padertorch.ops.sequence.mask import compute_mask
from padertorch.contrib.je.modules.conv import Pad
from padertorch.contrib.je.modules.features import NormalizedLogMelExtractor
from padertorch.contrib.je.modules.reduce import TakeLast, Mean
from padertorch.contrib.aw.transformer import TransformerEncoder
from padertorch.contrib.aw.patch_embed import pad_spec
from pb_sed.models import base
from paderbox.utils.nested import flatten, deflatten


class Transformer(base.SoundEventModel):
    """Transformer model for weakly supervised sound event detection.

    Parameters:
        feature_extractor: Feature extractor module
        patch_embed: Patch embedding module
        pos_enc: Positional encoding module
        encoder: Transformer encoder module
        predictor: Predictor module
        encoder_bwd: Transformer encoder module for backward direction
        predictor_bwd: Predictor module for backward direction
        share_weights_transformer: If True, the forward and backward transformer
            encoder share the same weights.
        share_weights_classifier: If True, the forward and backward predictor
            share the same weights.
        reverse_pos_enc_bwd: If True, the positional encoding of the backward
            direction is reversed.
        init_path: Initializer for the model parameters. If None, the default
            initializer of the model is used.

    >>> from padertorch.contrib.aw.predictor import PredictorHead
    >>> from padertorch.contrib.aw.patch_embed import PatchEmbed
    >>> from padertorch.contrib.aw.transformer import TransformerEncoder
    >>> from padertorch.contrib.aw.positional_encoding import SinCos2DPositionalEncoder
    >>> config = Transformer.get_config({\
            'encoder': {'factory': TransformerEncoder}, 'encoder_bwd': {'factory': TransformerEncoder},\
            'predictor': {'factory': PredictorHead, 768,
                 'num_classes': 10,
                 'classifier_hidden_dims':[]}, 'predictor_bwd': {'factory': PredictorHead, 768,
                 'num_classes': 10,
                 'classifier_hidden_dims':[],\},\
            'patch_embed': {'factory': PatchEmbed}, 'pos_enc': {'factory': SinCos2DPositionalEncoder},\
            'feature_extractor': {\
                'sample_rate': 16000,\
                'stft_size': 512,\
                'number_of_filters': 80,\
            },\
        })
    >>> transformer = Transformer.from_config(config)
    >>> np.random.seed(3)
    >>> inputs = {'stft': torch.tensor(np.random.randn(4, 1, 15, 257, 2), dtype=torch.float32), 'seq_len': [15, 14, 13, 12], 'weak_targets': torch.zeros((4,10)), 'boundary_targets': torch.zeros((4,10,15))}
    >>> outputs = transformer({**inputs})
    >>> outputs[0].shape
    torch.Size([4, 10, 15])
    >>> review = transformer.review(inputs, outputs)
    """
    def __init__(
            self, feature_extractor, patch_embed, pos_enc, encoder, predictor,
            *, encoder_bwd=None, predictor_bwd=None, share_weights_transformer=False, share_weights_classifier=False, minimum_score=1e-5, label_smoothing=0.,
            labelwise_metrics=(), label_mapping=None, test_labels=None,
            slat=False, strong_fwd_bwd_loss_weight=1., class_weights=None,
            reverse_pos_enc_bwd=False, init_path=None
    ):
        super().__init__(
            labelwise_metrics=labelwise_metrics,
            label_mapping=label_mapping,
            test_labels=test_labels,
        )
        self.feature_extractor = feature_extractor
        self.patch_embed = patch_embed
        self.pos_enc = pos_enc
        self.encoder = encoder
        if share_weights_transformer:
            # assert encoder_bwd is None
            self.encoder_bwd = self.encoder
        else:
            assert encoder_bwd is not None
            self.encoder_bwd = encoder_bwd
        self.predictor = predictor
        if share_weights_classifier:
            # assert predictor_bwd is None
            self.predictor_bwd = self.predictor
        else:
            assert predictor_bwd is not None
            self.predictor_bwd = predictor_bwd
        self.minimum_score = minimum_score
        self.label_smoothing = label_smoothing
        self.slat = slat
        self.strong_fwd_bwd_loss_weight = strong_fwd_bwd_loss_weight
        self.class_weights = None if class_weights is None else torch.Tensor(class_weights)
        self.reverse_pos_enc_bwd = reverse_pos_enc_bwd
        self.share_weights_transformer = share_weights_transformer

        self.initialize(init_path)
    
    def initialize(self, init_path):
        """Initialize the model parameters from BEATs.

        Args:
            init_path: Path to weight file.
        """
        if init_path is not None:
            pt_file = torch.load(init_path, map_location='cpu')
            if 'model' in pt_file:
                state_dict = pt_file['model']
            else:
                # assume all weights are saved in root
                state_dict = pt_file
            new_state_dict = {}
            for key, value in state_dict.items():
                if 'patch_embedding' in key:
                    key = key.replace('patch_embedding', 'patch_embed')
                if 'post_extract_proj' in key:
                    key = key.replace('post_extract_proj', 'patch_embed.out_proj')
                if 'pos_conv' in key:
                    key = key.replace('encoder.pos_conv.0', 'pos_enc.pos_conv')
                if key.startswith('layer_norm'):
                    key = key.replace('layer_norm', 'patch_embed.layer_norm')
                if key.startswith('encoder'):
                    if 'k_proj.weight' in key:
                        value = torch.cat((state_dict[key.replace('k_proj.weight', 'q_proj.weight')],
                                        value,
                                        state_dict[key.replace('k_proj.weight', 'v_proj.weight')]))
                        key = key.replace('k_proj.weight', 'qkv.weight')
                    if 'q_proj.weight' in key or 'v_proj.weight' in key:
                        continue
                    if 'q_proj.bias' in key or 'k_proj.bias' in key or 'v_proj.bias' in key:
                        key = key.replace('.bias', '')
                    if 'layers' in key:
                        key = key.replace('layers', 'blocks')
                    if 'fc1' in key:
                        key = key.replace('fc1', 'mlp.linear_0')
                    if 'fc2' in key:
                        key = key.replace('fc2', 'mlp.linear_1')
                    if 'self_attn' in key:
                        key = key.replace('self_attn', 'attn')
                    if 'self_attn_layer_norm' in key:
                        key = key.replace('attn_layer_norm', 'norm1')
                    if 'final_layer_norm' in key:
                        key = key.replace('final_layer_norm', 'norm2')
                    if 'relative_attention_bias' in key:
                        key = key.replace('relative_attention_bias', 'rel_pos_bias.relative_attention_bias')
                    if 'grep_a' in key:
                        key = key.replace('grep_a', 'rel_pos_bias.grep_w')
                    if 'grep_linear' in key:
                        key = key.replace('grep_linear', 'rel_pos_bias.grep_linear')
                new_state_dict[key] = value
                self.encoder.load_state_dict(new_state_dict, strict=False)
                if self.encoder_bwd is not None:
                    self.encoder_bwd.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded pretrained weights from {init_path}.")

    def sigmoid(self, y):
        """sigmoid function with minimum score margin
        
        Args:
            y: input
        
        Returns:
            sigmoid(y) with minimum score"""
        return self.minimum_score + (1-2*self.minimum_score) * nn.Sigmoid()(y)

    def reshape_to_grid(self, x, grid):
        """reshape patch-level predictions to grid
        
        Args:
            x: patch-level predictions, shape: (batch, num_patches, embedding_dim)
            grid: (grid_h, grid_w)
        
        Returns:
            x_grid, shape: (batch, grid_h, grid_w, embedding_dim)"""
        h, w = grid
        return rearrange(x, 'b (h w) d -> b h w d', h=h, w=w)
    
    def create_frame_level_predictions(self, y, pw=None, ow=None):
        """repeat patch-level predictions to frame-level predictions
        
        Args:
            y: patch-level predictions, shape: (batch, num_classes, grid_w)

        Returns:
            frame-level predictions, shape: (batch, num_classes, seq_len)
        """
        # y.shape: (batch, num_classes, grid_w)
        if pw is None:
            _, pw = self.patch_embed.patch_size
        if ow is None:
            _, ow = self.patch_embed.patch_overlap

        # repeat predictions to frame-level predictions
        if ow == 0:
            y = y.repeat_interleave(pw, dim=-1)
        else:
            y_reps = y.repeat_interleave(pw-ow, dim=-1)
            first = y[..., 0].unsqueeze(-1).expand(-1, -1, ow//2)
            last = y[..., -1].unsqueeze(-1).expand(-1, -1, ow//2)
            y = torch.cat((first, y_reps, last), dim=-1)
        
        return y
    
    def apply_encoder(self, x, grid, encoder):
        """apply encoder to patch sequence
        
        Args:
            x: patch sequence, shape: (batch, num_patches, embedding_dim)
            grid: (grid_h, grid_w)
            encoder: encoder module
        
        Returns:sigmoid
            y_grid: grid-level predictions, shape: (batch, grid_h, grid_w, embedding_dim)
            seq_len_y: sequence length of y_grid
        """
        y, seq_len_y = encoder(x, grid)  # output shape: (batch, num_patches, embedding_dim)
        y_grid = self.reshape_to_grid(y, grid=grid)  # output shape: (batch, grid_h, grid_w, embedding_dim)
        return y_grid, seq_len_y
    
    def apply_predictor(self, y_grid, predictor, pw=None, ow=None):
        """apply predictor to grid-level predictions

        Args:
            y_grid: grid-level predictions, shape: (batch, grid_h, grid_w, embedding_dim)
            predictor: predictor module
        
        Returns:
            probs: frame-level predictions, shape: (batch, num_classes, seq_len)
        """
        logits = predictor(y_grid)  # output shape: (batch, grid_w, num_classes)
        probs = self.sigmoid(logits)
        probs = rearrange(probs, 'b w c -> b c w')
        probs = self.create_frame_level_predictions(probs, pw=pw, ow=ow)
        return probs
    
    def fwd_tagging(self, feats, seq_len):
        seq_len_feats = feats.shape[-1]
        if feats.ndim == 3:
            feats = feats[:, None]
        assert feats.ndim == 4
        feats, pad_length = pad_spec(feats, self.patch_embed.patch_size, self.patch_embed.patch_overlap)
        x, grid = self.patch_embed(feats)
        x_fwd = self.pos_enc(x)
        y_fwd_grid, seq_len_y = self.apply_encoder(x_fwd, grid, self.encoder)
        probs_fwd = self.apply_predictor(y_fwd_grid, self.predictor)
        assert probs_fwd.shape[-1] == feats.shape[-1]
        if pad_length > 0:
            probs_fwd = probs_fwd[..., :-pad_length]
        assert probs_fwd.shape[-1] == seq_len_feats 
        return probs_fwd, seq_len

    def bwd_tagging(self, feats, seq_len):
        seq_len_feats = feats.shape[-1]
        if feats.ndim == 3:
            feats = feats[:, None]
        assert feats.ndim == 4
        feats, pad_length = pad_spec(feats, self.patch_embed.patch_size, self.patch_embed.patch_overlap)
        x, grid = self.patch_embed(feats)
        if self.reverse_pos_enc_bwd:
            x_bwd = self.revert_time(self.pos_enc(self.revert_time(x, grid)), grid)  # apply pos_enc to reversed input
        else:
            x_bwd = self.pos_enc(x)
        if self.share_weights_transformer:
            y_bwd_grid, seq_len_y_ = self.apply_encoder(self.revert_time(x_bwd, grid), grid, self.encoder_bwd)  # apply forward transformer to reversed input
            y_bwd_grid = y_bwd_grid.flip(-2)  # revert again for loss computation
        else:
            y_bwd_grid, seq_len_y_ = self.apply_encoder(x_bwd, grid, self.encoder_bwd)
        probs_bwd = self.apply_predictor(y_bwd_grid, self.predictor_bwd)
        assert probs_bwd.shape[-1] == feats.shape[-1]
        if pad_length > 0:
            probs_bwd = probs_bwd[..., :-pad_length]
        assert probs_bwd.shape[-1] == seq_len_feats 
        return probs_bwd, seq_len


    def revert_time(self, x, grid):
        """reverse time dimension
        
        Args:
            x: input, shape: (batch, p, d)
        
        Returns:
            x: input, shape: (batch, p, d)
        """
        x = rearrange(x, 'b (h w) d -> b h w d', h=grid[0], w=grid[1])
        x = x.flip(dims=[2])
        x = rearrange(x, 'b h w d -> b (h w) d')
        return x

    
    def forward(self, inputs):
        """
        forward used in trainer

        Args:
            inputs: example dict

        Returns:
            outputs: probs_fwd, probs_bwd, feats, targets, seq_len

        """
        # TODO: compute stft on GPU?
        if self.training:
            x = inputs.pop('stft')
        else:
            x = inputs['stft']
        seq_len = np.array(inputs['seq_len'])
        if "weak_targets" in inputs:
            targets = self.read_targets(inputs)
            feats, seq_len_x, targets = self.feature_extractor(
                x, seq_len=seq_len, targets=targets
            )
        else:
            feats, seq_len_x = self.feature_extractor(x, seq_len=seq_len)
            targets = None
        
        # feats, pad_length = pad_spec(feats, self.patch_embed.patch_size, self.patch_embed.patch_overlap)
        
        # x, grid = self.patch_embed(feats)
        # # x.shape: (batch, num_patches, embedding_dim)

        # x_fwd = self.pos_enc(x)
        
        # y_fwd_grid, seq_len_y = self.apply_encoder(x_fwd, grid, self.encoder)

        # probs_fwd = self.apply_predictor(y_fwd_grid, self.predictor)
        probs_fwd, seq_len_y = self.fwd_tagging(feats, seq_len_x)
        
        probs_bwd = None
        if self.encoder_bwd is None:
            probs_bwd = None
        else:
            # if self.reverse_pos_enc_bwd:
            #     x_bwd = self.revert_time(self.pos_enc(self.revert_time(x, grid)), grid)  # apply pos_enc to reversed input
            # else:
            #     x_bwd = x_fwd
            # if self.share_weights_transformer:
            #     y_bwd_grid, seq_len_y_ = self.apply_encoder(self.revert_time(x_bwd, grid), grid, self.encoder_bwd)  # apply forward transformer to reversed input
            #     y_bwd_grid = y_bwd_grid.flip(-2)  # revert again for loss computation
            # else:
            #     y_bwd_grid, seq_len_y_ = self.apply_encoder(x_bwd, grid, self.encoder_bwd)
            # probs_bwd = self.apply_predictor(y_bwd_grid, self.predictor_bwd)
            probs_bwd, seq_len_y_ = self.bwd_tagging(feats, seq_len_x)
            assert np.all(seq_len_y_ == seq_len_y)
    
        # probs.shape: (batch, num_classes, seq_len)

        # if pad_length > 0:
        #     feats = feats[..., :-pad_length]
        #     probs_fwd = probs_fwd[..., :-pad_length]
        #     probs_bwd = probs_bwd[..., :-pad_length]
        return probs_fwd, probs_bwd, feats, targets, np.asarray(inputs['seq_len'])

    def read_targets(self, inputs, subsample_idx=None):
        if 'boundary_targets' in inputs:
            return inputs['weak_targets'], inputs['boundary_targets']
        return inputs['weak_targets'],

    def review(self, inputs, outputs):
        """compute loss and metrics

        Args:
            inputs: example dict
            outputs: model outputs

        Returns:
            review: dict with loss and metrics
        """
        y_fwd, y_bwd, feats, targets, seq_len = outputs
        assert targets is not None
        weak_targets = targets[0]
        weak_targets_mask = (weak_targets < .01) + (weak_targets > .99)
        weak_targets = weak_targets * weak_targets_mask
        weak_label_rate = weak_targets_mask.detach().cpu().numpy().mean()
        
        loss = (
            self.compute_weak_fwd_bwd_loss(y_fwd, y_bwd, weak_targets, seq_len)
            * weak_targets_mask[..., None]
        )

        if self.strong_fwd_bwd_loss_weight > 0.:
            if self.slat:
                boundary_targets = weak_targets[..., None].expand(y_fwd.shape)
            else:
                assert len(targets) == 2, len(targets)
                boundary_targets = targets[1]
            boundary_targets_mask = (boundary_targets > .99) + (boundary_targets < .01)
            boundary_targets_mask = boundary_targets_mask * (boundary_targets_mask.float().mean(-1, keepdim=True) > .999) * (weak_targets > .99)[..., None]
            boundary_label_rate = boundary_targets_mask.detach().cpu().numpy().mean()
            if (boundary_targets_mask == 1).any():
                strong_label_loss = self.compute_strong_fwd_bwd_loss(
                y_fwd, y_bwd, boundary_targets)
                strong_fwd_bwd_loss_weight = (
                    boundary_targets_mask * self.strong_fwd_bwd_loss_weight)
                loss = strong_fwd_bwd_loss_weight * strong_label_loss + (1. - strong_fwd_bwd_loss_weight) * loss
        else:
            boundary_label_rate = 0.

        loss = Mean(axis=-1)(loss, seq_len)
        if self.class_weights is None:
            weights = weak_targets_mask
        else:
            self.class_weights = self.class_weights.to(loss.device)
            weights = weak_targets_mask * self.class_weights
        loss = (loss * weights).sum() / weights.sum()

        labeled_examples_idx = (
            weak_targets_mask.detach().cpu().numpy() == 1
        ).all(-1)
        y_weak = TakeLast(axis=2)(y_fwd, seq_len=seq_len)
        if y_bwd is not None:
            y_weak = y_weak / 2 + y_bwd[..., 0] / 2
        y_weak = y_weak.detach().cpu().numpy()[labeled_examples_idx]
        weak_targets = weak_targets.detach().cpu().numpy()[labeled_examples_idx]
        review = dict(
            loss=loss,
            scalars=dict(
                seq_len=np.mean(inputs['seq_len']),
                weak_label_rate=weak_label_rate,
                boundary_label_rate=boundary_label_rate,
            ),
            images=dict(
                features=feats[:3],
            ),
            buffers=dict(
                y_weak=y_weak,
                targets_weak=weak_targets,
            ),
        )
        return review
    
    def load_encoder_state_dict(self, state_dict):
        self.encoder.load_state_dict(
            flatten(state_dict['encoder']), strict=False)
        self.encoder_bwd.load_state_dict(
            flatten(state_dict['encoder_bwd']), strict=False)
        self.patch_embed.load_state_dict(
            flatten(state_dict['patch_embed']), strict=False)
        self.pos_enc.load_state_dict(
            flatten(state_dict['pos_enc']), strict=False)

    def compute_weak_fwd_bwd_loss(self, y_fwd, y_bwd, targets, seq_len):
        if self.label_smoothing > 0.:
            targets = torch.clip(
                targets, min=self.label_smoothing, max=1-self.label_smoothing)
        if y_bwd is None:
            y_weak = TakeLast(axis=2)(y_fwd, seq_len=seq_len)
            return nn.BCELoss(reduction='none')(y_weak, targets)[..., None].expand(y_fwd.shape)
        else:
            y_weak = torch.maximum(y_fwd, y_bwd)
            targets = targets[..., None].expand(y_weak.shape)
            return nn.BCELoss(reduction='none')(y_weak, targets)

    def compute_strong_fwd_bwd_loss(self, y_fwd, y_bwd, targets):
        if self.label_smoothing > 0.:
            targets = torch.clip(
                targets, min=self.label_smoothing, max=1-self.label_smoothing)
        strong_targets_fwd = torch.cummax(targets, dim=-1)[0]
        strong_targets_bwd = torch.cummax(targets.flip(-1), dim=-1)[0].flip(-1)
        # TODO: should we use logits instead? -> no sigmoid
        loss = nn.BCELoss(reduction='none')(y_fwd, strong_targets_fwd)
        if y_bwd is not None:
            loss = (
                loss/2
                + nn.BCELoss(reduction='none')(y_bwd, strong_targets_bwd)/2
            )
        return loss

    def modify_summary(self, summary):
        """called by the trainer before dumping a summary"""
        if f'targets_weak' in summary['buffers']:
            # Computes fscores from scores and targets
            self.add_metrics_to_summary(summary, 'weak')
        summary = super().modify_summary(summary)
        return summary

    def tagging(self, inputs):
        y_fwd, y_bwd, _, _, seq_len_y, *_ = self.forward(inputs)
        seq_len = torch.ones(seq_len_y.shape)
        if y_bwd is None:
            return TakeLast(axis=-1, keepdims=True)(y_fwd, seq_len_y), seq_len
        return (
            (
                TakeLast(axis=-1, keepdims=True)(y_fwd, seq_len_y)
                + y_bwd[..., :1]
            ) / 2,
            seq_len
        )

    def boundaries_detection(self, inputs):
        y_fwd, y_bwd, _, _, seq_len_y, *_ = self.forward(inputs)
        seq_mask = compute_mask(y_fwd, seq_len_y, batch_axis=0, sequence_axis=-1)
        return torch.minimum(y_fwd*seq_mask, y_bwd*seq_mask), seq_len_y

    def sound_event_detection(self, inputs, window_length, window_shift=1):
        """SED by applying the model to small segments around each frame

        Args:
            inputs:
            window_length:
            window_shift:

        Returns:

        """
        window_length = np.array(window_length, dtype=int)
        x = inputs['stft']
        seq_len = np.array(inputs['seq_len'])
        feats, seq_len = self.feature_extractor(x, seq_len=seq_len)


        if window_length.ndim == 0:
            return self._single_window_length_sed(
                feats, seq_len, window_length, window_shift
            )
        window_lengths_flat = np.unique(window_length.flatten())
        y = None
        for i, win_len in enumerate(window_lengths_flat):
            yi, seq_len_y = self._single_window_length_sed(
                feats, seq_len, win_len, window_shift
            )
            b, k, t = yi.shape
            if window_length.ndim == 1:
                assert window_length.shape[0] in [1, k], window_length.shape
            elif window_length.ndim == 2:
                assert window_length.shape[1] in [1, k], window_length.shape
                n = window_length.shape[0]
                window_length = np.broadcast_to(window_length, (n, k))
                yi = yi[:, None]
            else:
                raise ValueError(
                    'window_length.ndim must not be greater than 2.')
            if y is None:
                y = torch.zeros((b, *window_length.shape, t), device=yi.device)
            mask = torch.from_numpy(window_length.copy()).to(yi.device) == win_len
            y += mask[..., None] * yi
        return y, seq_len_y

    def _single_window_length_sed(
            self, feats, seq_len, window_length, window_shift
    ):
        b, _, f, t = feats.shape
        h = rearrange(feats, 'b 1 f t -> b f t')
        if window_length > window_shift:
            h = Pad('both')(h, (window_length - window_shift))
        h = Pad('end')(h, window_shift - 1)
        indices = np.arange(0, t, window_shift)
        h = [h[..., i:i + window_length] for i in indices]
        n = len(h)
        h = torch.cat(h, dim=0)
        y, _ = self.fwd_tagging(h, seq_len=None)
        y = rearrange(y[..., -1], '(n b) k -> b k n', b=b, n=n)
        if self.encoder_bwd is not None:
            y_bwd, _ = self.bwd_tagging(h, seq_len=None)
            y_bwd = rearrange(y_bwd[..., 0], '(n b) k -> b k n', b=b, n=n)
            y = (y + y_bwd) / 2
        seq_len = 1 + (seq_len-1) // window_shift
        return y, seq_len

    @classmethod
    def finalize_dogmatic_config(cls, config):
        """Automatically prepares/completes the configuration of the model.

        You do not need to understand how this is working as there is a lot of
        magic in the background which serves convenience and is not crucial to
        run the model.

        Args:
            config:

        Returns:

        """
        config['feature_extractor'] = {'factory': NormalizedLogMelExtractor}
        input_size = config['feature_extractor']['number_of_filters']


def tune_tagging(
        models, dataset, device, timestamps, event_classes, metrics,
        minimize=False, storage_dir=None
):
    print()
    print('Tagging Tuning')
    tagging_scores = base.tagging(
        models, dataset, device,
        timestamps=timestamps, event_classes=event_classes,
    )
    return base.tune_tagging(
        tagging_scores, medfilt_length_candidates=[1],
        metrics=metrics, minimize=minimize, storage_dir=storage_dir,
    )


def tune_boundary_detection(
        models, dataset, device, timestamps, event_classes, tags, metrics,
        stepfilt_lengths, minimize=False, tag_masking='?', storage_dir=None,
):
    print()
    print('Boundaries Detection Tuning')
    boundaries_scores = base.boundaries_detection(
        models, dataset, device,
        stepfilt_length=None, apply_mask=False, masks=tags,
        timestamps=timestamps, event_classes=event_classes,
    )
    return base.tune_boundaries_detection(
        boundaries_scores, medfilt_length_candidates=[1],
        stepfilt_length_candidates=stepfilt_lengths,
        tags=tags, metrics=metrics, minimize=minimize,
        tag_masking=tag_masking,
        storage_dir=storage_dir,
    )


def tune_sound_event_detection(
        models, dataset, device, timestamps, event_classes, tags, metrics,
        window_lengths, window_shift, medfilt_lengths,
        minimize=False, tag_masking='?', storage_dir=None,
):
    print()
    print('Sound Event Detection Tuning')
    leaderboard = {}
    for win_len in window_lengths:
        print()
        print(f'### window_length={win_len} ###')
        detection_scores = base.sound_event_detection(
            models, dataset, device,
            model_kwargs={
                'window_length': win_len, 'window_shift': window_shift
            },
            timestamps=timestamps[::window_shift], event_classes=event_classes,
        )
        leaderboard_for_winlen = base.tune_sound_event_detection(
            detection_scores, medfilt_lengths, tags,
            metrics=metrics, minimize=minimize,
            tag_masking=tag_masking,
            storage_dir=storage_dir,
        )
        for metric_name in leaderboard_for_winlen:
            metric_values = leaderboard_for_winlen[metric_name][0]
            hyper_params = leaderboard_for_winlen[metric_name][1]
            scores = leaderboard_for_winlen[metric_name][2]
            for event_class in event_classes:
                hyper_params[event_class]['window_length'] = win_len
                hyper_params[event_class]['window_shift'] = window_shift
            leaderboard = base.update_leaderboard(
                leaderboard, metric_name, metric_values, hyper_params, scores,
                minimize=minimize,
            )
    print()
    print('best overall:')
    for metric_name in metrics:
        print()
        print(metric_name, ':')
        print(leaderboard[metric_name][0])

    return leaderboard
