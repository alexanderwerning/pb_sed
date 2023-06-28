from pb_sed.models.weak_label.transformer import Transformer
import torch.nn as nn
import torch
from padertorch.contrib.aw.patch_embed import pad_spec
import numpy as np
from einops import rearrange
from paderbox.utils.random_utils import str_to_seed

class SplitTransformer(Transformer):
    def __init__(self, *args, seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.encoder_bwd is None or self.encoder == self.encoder_bwd
        assert self.predictor_bwd is None or self.predictor == self.predictor_bwd
        assert self.encoder.use_cls_token
        self.strong_fwd_bwd_loss_weight = 0
        self.seed = seed
        self.random_generator = np.random.RandomState(seed=seed)
    
    def fwd_tagging(self, feats, seq_len):
        """Return a single probability vector per example."""
        seq_len_feats = feats.shape[-1]
        if feats.ndim == 3:
            feats = feats[:, None]
        assert feats.ndim == 4
        feats, pad_length = pad_spec(feats, self.patch_embed.patch_size, self.patch_embed.patch_overlap)
        x, grid = self.patch_embed(feats)
        x_fwd = self.pos_enc(x)
        y_fwd, seq_len_y = self.encoder(x)
        class_token = y_fwd[..., None, 0, None, :]  # shape [B x H=1 x W=1 x D]
        prediction = self.predictor(class_token)  # use class token
        probs = nn.functional.sigmoid(prediction)
        probs = rearrange(probs, 'b w c -> b c w')
        seq_len = np.ones_like(seq_len)
        return probs, seq_len

    def bwd_tagging(self, feats, seq_len):
        """Same as fwd_tagging, needed for compatibility."""
        return self.fwd_tagging(feats, seq_len)
    
    def sample_split_index(self, seq_len):
        return self.random_generator.randint(1, np.max(seq_len)-2)
        return np.random.randint(1, np.max(seq_len)-2)
    
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
    
        # fix seed for reproducibility
        batch_id = "".join(inputs["example_id"])
        seed = str_to_seed((str(self.seed) + batch_id))
        self.random_generator.seed(seed)

        if not self.training:
            split_index = feats.shape[-1]//2  # TODO: 50/50 split or 100/0 split, deterministic random fn?
        else:
            split_index = self.sample_split_index(feats.shape[-1])
        # feats_list = [feats[..., :split_index], feats[..., split_index:]]

        def printGPUInfo(verbose=False):
            print("total memory available: " + str(torch.cuda.get_device_properties(0).total_memory * 1e-9))
            print("Currently reserved: " + str(torch.cuda.memory_reserved(0) * 1e-9))
            print("Currently allocated: " + str(torch.cuda.memory_allocated(0) * 1e-9))
            print("Max reserved " + str(torch.cuda.max_memory_reserved(0) * 1e-9))
            print("Max allocated " + str(torch.cuda.max_memory_reserved(0) * 1e-9))
            print("\n")
            if verbose:
                print(torch.cuda.memory_summary(device=None, abbreviated=False))

        probs_fwd, seq_len_y = self.fwd_tagging(feats[..., :split_index], seq_len_x)
        # probs shape: [B x C x W=1]
        # probs_fwd = torch.rand((x.shape[0], 10, 1)).to(x.device)
        assert probs_fwd.shape[-1] == 1
        seq_len_y = np.ones_like(seq_len)
        probs_bwd = torch.rand((x.shape[0], 10, 1)).to(x.device)
        printGPUInfo(verbose=False)
        breakpoint()
        return probs_fwd, probs_bwd, feats, targets, seq_len_y
