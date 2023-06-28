from pb_sed.models.weak_label.transformer import Transformer
import torch.nn as nn
import torch
from padertorch.contrib.aw.patch_embed import pad_spec
import numpy as np

class TransformerSlices(Transformer):
    def __init__(self, *args, cnn_config=2*((2,2), (2,2)), **kwargs):
        super().__init__(*args, **kwargs)
        # downsampling
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(1, 1, kernel_size=(2, 2), stride=(2, 2)),
            )
    
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
        
        feats, pad_length  = pad_spec(feats, (None, 4), (None, 4))

        # downsampling
        feats = self.cnn(feats)
        
        x, grid = self.patch_embed(feats)
        # x.shape: (batch, num_patches, embedding_dim)

        x_fwd = self.pos_enc(x)
        
        #TODO: mask padding, sequence length is padded, what happens to the target etc?
        y_fwd_grid, seq_len_y = self.apply_encoder(x_fwd, grid, self.encoder)

        probs_fwd = self.apply_predictor(y_fwd_grid, self.predictor, pw=4, ow=0)
        
        probs_bwd = None
        if self.encoder_bwd is None:
            probs_bwd = None
        else:
            if self.reverse_pos_enc_bwd:
                x_bwd = self.revert_time(self.pos_enc(self.revert_time(x, grid)), grid)  # apply pos_enc to reversed input
            else:
                x_bwd = x_fwd
            if self.share_weights_transformer:
                y_bwd_grid, seq_len_y_ = self.apply_encoder(self.revert_time(x_bwd, grid), grid, self.encoder_bwd)  # apply forward transformer to reversed input
                y_bwd_grid = y_bwd_grid.flip(-2)  # revert again for loss computation
            else:
                y_bwd_grid, seq_len_y_ = self.apply_encoder(x_bwd, grid, self.encoder_bwd)
            probs_bwd = self.apply_predictor(y_bwd_grid, self.predictor_bwd, pw=4, ow=0)
            assert (seq_len_y_ == seq_len_y)

        # probs.shape: (batch, num_classes, seq_len)
        if pad_length > 0:
            probs_fwd = probs_fwd[..., :-pad_length]
            probs_bwd = probs_bwd[..., :-pad_length]
        return probs_fwd, probs_bwd, feats, targets, np.asarray(inputs['seq_len'])