import torch
import torch.nn as nn


class NewestSketch(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.num_sketch_tokens = model_config.num_sketch_tokens

    def forward(self, kv, sketch):
        """
        kv.shape [batch_size, num_heads, seq_len, head_size]
        sketch.shape [batch_size, num_heads, num_sketch_tokens, head_size]
        """
        return kv[:, :, :self.num_sketch_tokens, :]


class FrequentDirections(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.num_sketch_tokens = model_config.num_sketch_tokens

    def forward(self, kv, sketch):
        """
        kv.shape [batch_size, num_heads, seq_len, head_size]
        sketch.shape [batch_size, num_heads, num_sketch_tokens, head_size]
        """
        if sketch is None:
            return kv
        x = torch.cat([sketch, kv], dim=-2).to(torch.float32)
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        l = self.num_sketch_tokens
        delta = S[..., l:l+1] ** 2 if l < S.shape[-1] else 0
        S_new = torch.sqrt(torch.clamp(S[..., :l] ** 2 - delta, min=0.0))
        B = S_new.unsqueeze(-1) * Vh[..., :l, :]
        B = B.to(kv.dtype)
        return B


def make_sketch(model_config):
    if model_config.sketch_type == "newest":
        return NewestSketch(model_config)
    elif model_config.sketch_type == "frequent_directions":
        return FrequentDirections(model_config)
    else:
        raise NotImplementedError("unsupported sketch type {}".format(model_config.sketch_type))


if __name__ == "__main__":
    kv = torch.randn(1, 32, 728, 4096, dtype=torch.bfloat16, device=0)
    sketch_kv = torch.randn(1, 32, 728, 4096, dtype=torch.bfloat16, device=0)

    class ModelConfig:
        def __init__(self):
            self.num_sketch_tokens = 728
    model_config = ModelConfig()
    sketch = FrequentDirections(model_config)
    sketch_kv = sketch(kv, sketch_kv)
    breakpoint()
