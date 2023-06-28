from pb_sed.models.weak_label.transformer import Transformer
import torch.nn as nn
import torch
from padertorch.contrib.aw.patch_embed import pad_spec
import numpy as np
from einops import rearrange
from paderbox.utils.random_utils import str_to_seed
from pb_sed.models import base

class SplitTransformer(base.SoundEventModel):
    def __init__(self, model, feature_extractor, splits=1, seed=None):
        super().__init__()
        self.model = model
        self.feature_extractor = feature_extractor
        self.seed = seed
        self.random_generator = np.random.RandomState(seed)
        self.splits = splits
    
    def fwd_tagging(self, feats, seq_len):
        """Return a single probability vector per example."""
        seq_len_feats = feats.shape[-1]
        if feats.ndim == 3:
            feats = feats[:, None]
        assert feats.ndim == 4
        weak_predictions, strong_predictions = self.model.tagging(feats, seq_len_feats)
        return weak_predictions, seq_len

    def bwd_tagging(self, feats, seq_len):
        """Same as fwd_tagging, needed for compatibility."""
        return self.fwd_tagging(feats, seq_len)
    
    def sample_split_index(self, seq_len):
        return self.random_generator.randint(1, np.max(seq_len)-2)
    
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
            targets = self.model.read_targets(inputs)
            feats, seq_len_x, targets = self.feature_extractor(
                x, seq_len=seq_len, targets=targets
            )
        else:
            feats, seq_len_x = self.feature_extractor(x, seq_len=seq_len)
            targets = None
    
        # fix seed for reproducibility
        batch_id = "".join(inputs["example_id"])
        seed = str_to_seed((str(self.seed) + batch_id))
        self.random_generator.seed(seed)

        if self.splits == 1:
            if not self.training:
                split_index = feats.shape[-1]//2  # TODO: 50/50 split or 100/0 split, deterministic random fn?
            else:
                split_index = 16+self.sample_split_index(feats.shape[-1]-32)

            probs_fwd, seq_len_y = self.fwd_tagging(feats[..., :split_index], seq_len_x)
            # probs shape: [B x C x W=1]
            # probs_fwd = torch.rand((x.shape[0], 10, 1)).to(x.device)
            
            probs_bwd, _ = self.fwd_tagging(feats[..., split_index:], seq_len_x)
            # return probs_fwd, probs_bwd, feats, targets, seq_len_y
            weak_predictions = torch.maximum(probs_fwd, probs_bwd)
        elif self.splits == 0:
            weak_predictions, _ = self.fwd_tagging(feats, seq_len_x)
        else:
            raise NotImplementedError("Not implemented yet")
        seq_len_y = np.ones_like(seq_len)
        return weak_predictions, None, feats, targets, seq_len_y

    def review(self, inputs, outputs):
        weak_predictions, _, feats, targets, seq_len = outputs
        assert targets is not None
        weak_targets = targets[0]
        weak_targets_mask = (weak_targets < 0.01) + (weak_targets > 0.99)
        weak_targets = weak_targets * weak_targets_mask
        weak_label_rate = weak_targets_mask.detach().cpu().numpy().mean()

        loss = (
            self.model.compute_weak_loss(weak_predictions, weak_targets, seq_len)
            * weak_targets_mask
        )
       
        boundary_label_rate = 0.0

        weights = weak_targets_mask

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

    def modify_summary(self, summary):
        if f"targets_weak" in summary["buffers"]:
            # Computes fscores from scores and targets
            self.add_metrics_to_summary(summary, "weak")
        summary = super().modify_summary(summary)
        return summary

    def sound_event_detection(self, inputs, window_length, window_shift=1):
        return self.model.sound_event_detection(inputs, window_length, window_shift)
    
    def tagging(self, feats, seq_len):
        return self.model.tagging(feats, seq_len)
    
    def boundaries_detection(self, inputs, **params):
        raise NotImplementedError("Not implemented yet")