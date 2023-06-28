import numpy as np
import torch
from padertorch.contrib.je.modules.reduce import Mean
from torch import nn
from padertorch.contrib.je.modules.conv import Pad
from einops import rearrange
from padertorch.contrib.aw.patch_embed import pad_spec

from padertorch.contrib.je.modules.features import NormalizedLogMelExtractor
from padertorch import Model
from pb_sed.models import base

def create_frame_level_predictions(y, pw=16, ow=0):
    """repeat patch-level predictions to frame-level predictions
    
    Args:
        y: patch-level predictions, shape: (batch, num_classes, grid_w)

    Returns:
        frame-level predictions, shape: (batch, num_classes, seq_len)
    """
    # y.shape: (batch, num_classes, grid_w)

    # repeat predictions to frame-level predictions
    if ow == 0:
        y = y.repeat_interleave(pw, dim=-1)
    else:
        y_reps = y.repeat_interleave(pw-ow, dim=-1)
        first = y[..., 0].unsqueeze(-1).expand(-1, -1, ow//2)
        last = y[..., -1].unsqueeze(-1).expand(-1, -1, ow//2)
        y = torch.cat((first, y_reps, last), dim=-1)
    
    return y

class ViTPredictor(nn.Module):
    def __init__(self, grid_h, embed_dim, num_classes, use_class_token=False):
        super().__init__()
        self.grid_h = grid_h
        self.num_classes = num_classes
        self.weak_classifier = nn.Linear(embed_dim, num_classes)
        self.strong_classifier = nn.Linear(embed_dim, num_classes)
        self.use_class_token = use_class_token

    def forward(self, x):
        # x.shape = (batch_size, num_patches, embed_dim)
        if self.use_class_token:
            cls_token = x[:, 0]
            weak_logits = self.weak_classifier(cls_token)
            weak_probs = torch.sigmoid(weak_logits)
            sequence = x[:, 1:]
            strong_logits = self.strong_classifier(sequence)
        else:
            strong_logits = self.strong_classifier(x)
            weak_logits = torch.mean(strong_logits, dim=1)
            weak_probs = torch.sigmoid(weak_logits)
            sequence = x
        B, N, D = strong_logits.shape
        patchwise_strong_logits_grid = strong_logits.reshape(B, self.grid_h, -1, self.num_classes)
        patchwise_strong_mean_logits = torch.mean(patchwise_strong_logits_grid, dim=1)
        patchwise_strong_probs = torch.sigmoid(patchwise_strong_mean_logits)
        patchwise_strong_probs = rearrange(patchwise_strong_probs, 'b w c -> b c w')
        strong_probs = create_frame_level_predictions(patchwise_strong_probs)
        return weak_probs, strong_probs

class SEDModel(base.SoundEventModel):
    def __init__(self,
                 feature_extractor,
                 encoder, 
                 predictor,
                 strong_loss_weight=1.,
                 label_smoothing=0.,
                 slat=False,
                 class_weights=None,):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.predictor = predictor
        self.strong_loss_weight = strong_loss_weight
        self.label_smoothing = label_smoothing
        self.slat = slat
        self.class_weights = None if class_weights is None else torch.Tensor(class_weights)
        self.padding_params = (16, 16), (0,0)

    def forward(self, inputs):
        if self.training:
            x = inputs.pop("stft")
        else:
            x = inputs["stft"]
        seq_len = np.array(inputs["seq_len"])
        if "weak_targets" in inputs:
            targets = self.read_targets(inputs)
            feats, seq_len_x, targets = self.feature_extractor(
                x, seq_len=seq_len, targets=targets
            )
        else:
            feats, seq_len_x = self.feature_extractor(x, seq_len=seq_len)
            targets = None
        feats, pad_length = pad_spec(feats, self.padding_params[0], self.padding_params[1])

        h = self.encoder(feats)

        weak_predictions, strong_predictions = self.predictor(h)
        if pad_length > 0:
            strong_predictions = strong_predictions[..., :-pad_length]
        return weak_predictions, strong_predictions, feats, targets, seq_len_x

    def tagging(self, feats, seq_len):
        h = self.encoder(feats)
        weak_predictions, strong_predictions = self.predictor(h)
        return weak_predictions, strong_predictions

    def review(self, inputs, outputs):
        weak_predictions, strong_predictions, feats, targets, seq_len = outputs
        assert targets is not None
        weak_targets = targets[0]
        weak_targets_mask = (weak_targets < 0.01) + (weak_targets > 0.99)
        weak_targets = weak_targets * weak_targets_mask
        weak_label_rate = weak_targets_mask.detach().cpu().numpy().mean()

        loss = (
            self.compute_weak_loss(weak_predictions, weak_targets, seq_len)
            * weak_targets_mask
        )

        # required by Mean later
        loss = loss[..., None]

        boundary_label_rate = 0.0
        if self.strong_loss_weight > 0.0 and (self.slat or len(targets) == 2):
            loss = loss.expand(strong_predictions.shape)
            if self.slat:  # use weak targets as boundary targets
                boundary_targets = weak_targets[..., None].expand(
                    strong_predictions.shape
                )
            else:
                boundary_targets = targets[1]
            boundary_targets_mask = (boundary_targets > 0.99) + (
                boundary_targets < 0.01
            )
            boundary_targets_mask = (
                boundary_targets_mask
                * (boundary_targets_mask.float().mean(-1, keepdim=True) > 0.999)
                * (weak_targets > 0.99)[..., None]
            )
            boundary_label_rate = boundary_targets_mask.detach().cpu().numpy().mean()
            if (boundary_targets_mask == 1).any():
                strong_label_loss = self.compute_strong_loss(
                    strong_predictions, boundary_targets
                )
                strong_loss_weight = (
                    boundary_targets_mask * self.strong_loss_weight
                )
                loss = (
                    strong_loss_weight * strong_label_loss
                    + (1.0 - strong_loss_weight) * loss
                )
        elif self.strong_loss_weight == 0:
            boundary_label_rate = 0.0
        elif len(targets) == 1:
            # print("No strong labels provided")
            pass

        loss = Mean(axis=-1)(loss, seq_len)
        if self.class_weights is None:
            weights = weak_targets_mask
        else:
            self.class_weights = self.class_weights.to(loss.device)
            weights = weak_targets_mask * self.class_weights

        loss = (loss * weights).sum() / weights.sum()

        labeled_examples_idx = (weak_targets_mask.detach().cpu().numpy() == 1).all(-1)

        y_weak = weak_predictions.detach().cpu().numpy()[labeled_examples_idx]
        weak_targets = weak_targets.detach().cpu().numpy()[labeled_examples_idx]
        review = dict(
            loss=loss,
            scalars=dict(
                seq_len=np.mean(inputs["seq_len"]),
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

    def compute_weak_loss(self, weak_predictions, targets, seq_len):
        if self.label_smoothing > 0.0:
            targets = torch.clip(
                targets, min=self.label_smoothing, max=1 - self.label_smoothing
            )
        return nn.BCELoss(reduction="none")(weak_predictions, targets)

    def compute_strong_loss(self, strong_predictions, targets):
        if self.label_smoothing > 0.0:
            targets = torch.clip(
                targets, min=self.label_smoothing, max=1 - self.label_smoothing
            )
        return nn.BCELoss(reduction="none")(strong_predictions, targets)

    def modify_summary(self, summary):
        if f"targets_weak" in summary["buffers"]:
            # Computes fscores from scores and targets
            self.add_metrics_to_summary(summary, "weak")
        summary = super().modify_summary(summary)
        return summary

    def boundaries_detection(self, inputs):
        feats, strong_predictions, weak_predictions, targets, seq_len = self.forward(
            inputs
        )
        return strong_predictions

    def sound_event_detection(self, inputs, window_length, window_shift=1):
        """SED by applying the model to small segments around each frame"""
        window_length = np.array(window_length, dtype=int)
        x = inputs["stft"]
        seq_len = np.array(inputs["seq_len"])
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
                raise ValueError("window_length.ndim must not be greater than 2.")
            if y is None:
                y = torch.zeros((b, *window_length.shape, t), device=yi.device)
            mask = torch.from_numpy(window_length.copy()).to(yi.device) == win_len
            y += mask[..., None] * yi
        return y, seq_len_y
    
    def read_targets(self, inputs, subsample_idx=None):
        if 'boundary_targets' in inputs:
            return inputs['weak_targets'], inputs['boundary_targets']
        return inputs['weak_targets'],

    def _single_window_length_sed(self, feats, seq_len, window_length, window_shift):
        b, _, f, t = feats.shape
        h = rearrange(feats, "b 1 f t -> b f t")
        if window_length > window_shift:
            h = Pad("both")(h, (window_length - window_shift))
        h = Pad("end")(h, window_shift - 1)
        indices = np.arange(0, t, window_shift)
        h = [h[..., i : i + window_length] for i in indices]
        n = len(h)
        h = torch.cat(h, dim=0)
        y, _ = self.fwd_tagging(h, seq_len=None)
        y = rearrange(y[..., -1], "(n b) k -> b k n", b=b, n=n)
        
        seq_len = 1 + (seq_len - 1) // window_shift
        return y, seq_len

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config["feature_extractor"] = {"factory": NormalizedLogMelExtractor}
        input_size = config["feature_extractor"]["number_of_filters"]
