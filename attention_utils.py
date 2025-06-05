import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn_kv_score import flash_attn_kv_score

# def select_offload_kv(query_states, key_states, value_states, image_mask, top_k):
#     # query_states.shape [1, num_heads, seq_len, head_dim]
#     # key_states.shape [1, num_heads, seq_len, head_dim]
#     # value_states.shape [1, num_heads, seq_len, head_dim]
#     bsz, num_heads, seq_len, head_dim = query_states.shape
#     device = query_states.device
#     if bsz != 1:
#         raise NotImplementedError("only support batch size 1")

#     #seq_len = 22000
#     query_states = query_states.squeeze(0)#[:, :seq_len, :]
#     key_states = key_states.squeeze(0)#[:, :seq_len, :]
#     value_states = value_states.squeeze(0)#[:, :seq_len, :]

#     attention_scores = torch.bmm(
#         key_states,  # [num_heads, seq_len, head_dim]
#         query_states.transpose(1, 2)  # [num_heads, head_dim, seq_len]
#     )  # [num_heads, seq_len, seq_len]
#     attention_scores = F.softmax(attention_scores, dim=2)
#     attention_scores = attention_scores.sum(dim=1)  # [num_heads, seq_len]
#     _, indices = torch.topk(attention_scores, k=top_k, dim=1)  # [num_heads, top_k]
#     del attention_scores
    
#     indices_expanded = indices.unsqueeze(-1).expand(-1, -1, head_dim)  # [num_heads, top_k, head_dim]
#     gpu_key_states = torch.gather(key_states, 1, indices_expanded)  # [num_heads, top_k, head_dim]
#     gpu_value_states = torch.gather(value_states, 1, indices_expanded)  # [num_heads, top_k, head_dim]

#     mask = torch.zeros(num_heads, seq_len, dtype=torch.bool, device=device)
#     mask.scatter_(1, indices, True)  # [num_heads, seq_len]
#     remaining_mask = ~mask  # [num_heads, seq_len]
#     remaining_count = seq_len - top_k

#     cpu_key_states = key_states[remaining_mask].view(num_heads, remaining_count, head_dim)
#     cpu_value_states = value_states[remaining_mask].view(num_heads, remaining_count, head_dim)

#     return (
#         gpu_key_states.unsqueeze(0),  # [1, num_heads, top_k, head_dim]
#         gpu_value_states.unsqueeze(0),  # [1, num_heads, top_k, head_dim]
#         cpu_key_states.unsqueeze(0),  # [1, num_heads, remaining_count, head_dim]
#         cpu_value_states.unsqueeze(0),  # [1, num_heads, remaining_count, head_dim]
#     )


class SelectOffloadKV(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.top_p = model_config.top_p

    def forward(self, query_states, key_states, value_states, image_mask):
        # query_states.shape [1, num_heads, seq_len, head_dim]
        # key_states.shape [1, num_heads, seq_len, head_dim]
        # value_states.shape [1, num_heads, seq_len, head_dim]
        # image_mask.shape [1, seq_len]
        bsz, num_heads, seq_len, head_dim = query_states.shape
        device = query_states.device
        if bsz != 1:
            raise NotImplementedError("only support batch size 1")
        
        image_mask = image_mask.squeeze(0)
        image_keys = key_states[0, :, image_mask]  # [num_heads, num_image_tokens, head_dim]
        image_values = value_states[0, :, image_mask]  # [num_heads, num_image_tokens, head_dim]
        num_image_tokens = image_keys.shape[1]
        top_k = int(num_image_tokens * self.top_p)
        if top_k > num_image_tokens:
            raise ValueError("top_k is greater than the number of image tokens")
        
        last_image_pos = torch.where(image_mask)[0][-1].item()
        if last_image_pos >= seq_len:
            raise ValueError("no text tokens found after image")
        text_queries = query_states[0, :, last_image_pos+1:]  # [num_heads, num_text_tokens, head_dim]

        attention_scores = flash_attn_kv_score(
            text_queries.unsqueeze(0).transpose(1, 2),
            image_keys.unsqueeze(0).transpose(1, 2)
        ).squeeze(0)  # [num_heads, num_image_tokens]
        _, indices = torch.topk(attention_scores, k=top_k, dim=1)  # [num_heads, top_k]
        del attention_scores
        
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, head_dim)  # [num_heads, top_k, head_dim]
        top_image_keys = torch.gather(image_keys, 1, indices_expanded)  # [num_heads, top_k, head_dim]
        top_image_values = torch.gather(image_values, 1, indices_expanded)  # [num_heads, top_k, head_dim]

        mask = torch.zeros(num_heads, num_image_tokens, dtype=torch.bool, device=device)
        mask.scatter_(1, indices, True)  # [num_heads, num_image_tokens]

        remaining_mask = ~mask  # [num_heads, num_image_tokens]
        remaining_count = num_image_tokens - top_k

        cpu_key_states = image_keys[remaining_mask].view(num_heads, remaining_count, head_dim)
        cpu_value_states = image_values[remaining_mask].view(num_heads, remaining_count, head_dim)

        text_mask = ~image_mask
        text_keys = key_states[0, :, text_mask]
        text_values = value_states[0, :, text_mask]
        gpu_key_states = torch.cat([text_keys, top_image_keys], dim=1).unsqueeze(0)
        gpu_value_states = torch.cat([text_values, top_image_values], dim=1).unsqueeze(0)

        return (
            gpu_key_states,  # [1, num_heads, top_k, head_dim]
            gpu_value_states,  # [1, num_heads, top_k, head_dim]
            cpu_key_states.unsqueeze(0),  # [1, num_heads, remaining_count, head_dim]
            cpu_value_states.unsqueeze(0),  # [1, num_heads, remaining_count, head_dim]
        )


# class SelectOffloadKV(nn.Module):
#     def __init__(self, model_config):
#         super().__init__()
#         self.top_p = model_config.top_p
#         self.n_hashes = model_config.n_hashes
#         self.n_bits = model_config.n_bits
#         self.bucket_size = 2 ** model_config.n_bits
#         self.head_dim = model_config.head_dim

#         self.register_buffer(
#             "rotations",
#             torch.randn(self.head_dim, self.n_hashes, self.n_bits) * 0.2
#         )
#         self.register_buffer(
#             "binary_pack",
#             2 ** torch.arange(self.n_bits)
#         )

#     def forward(self, query_states, key_states, value_states, image_mask):
#         # query_states.shape [1, num_heads, seq_len, head_dim]
#         # key_states.shape [1, num_heads, seq_len, head_dim]
#         # value_states.shape [1, num_heads, seq_len, head_dim]
#         # image_mask.shape [1, seq_len]
#         bsz, num_heads, seq_len, head_dim = query_states.shape
#         top_k = int(seq_len * self.top_p)
#         device = query_states.device
#         if bsz != 1:
#             raise NotImplementedError("only support batch size 1")

#         image_mask = image_mask.squeeze(0)
#         image_keys = key_states[0, :, image_mask]  # [num_heads, num_image_tokens, head_dim]
#         image_values = value_states[0, :, image_mask]  # [num_heads, num_image_tokens, head_dim]
#         num_image_tokens = image_keys.shape[1]
#         if top_k > num_image_tokens:
#             raise ValueError("top_k is greater than the number of image tokens")
        
#         last_image_pos = torch.where(image_mask)[0][-1].item()
#         if last_image_pos >= seq_len:
#             raise ValueError("no text tokens found after image")
#         text_queries = query_states[0, :, last_image_pos+1:]  # [num_heads, num_text_tokens, head_dim]

#         centered_image_keys = image_keys - image_keys.mean(dim=-1, keepdim=True)
#         key_hashes = torch.einsum("nid,dhb->nihb", centered_image_keys, self.rotations) > 0
#         key_hashes = (key_hashes * self.binary_pack).sum(-1)  # [num_heads, num_image_tokens, n_hashes]
#         key_hashes = key_hashes.permute(0, 2, 1)  # [num_heads, n_hashes, num_image_tokens]

#         centered_text_queries = text_queries - text_queries.mean(dim=-1, keepdim=True)
#         query_hashes = torch.einsum("ntd,dhb->nthb", centered_text_queries, self.rotations) > 0
#         query_hashes = (query_hashes * self.binary_pack).sum(-1)  # [num_heads, num_text_tokens, n_hashes]
#         query_hashes = query_hashes.permute(0, 2, 1)  # [num_heads, n_hashes, num_text_tokens]

#         query_buckets = torch.zeros(
#             [num_heads, self.n_hashes, self.bucket_size],
#             dtype=torch.long,
#             device=device
#         )
#         query_buckets.scatter_add_(
#             dim=2,
#             index=query_hashes,
#             src=torch.ones_like(query_hashes, dtype=torch.long, device=device)
#         )

#         gathered = torch.gather(query_buckets, dim=2, index=key_hashes)
#         attention_scores = gathered.sum(dim=1)

#         _, indices = torch.topk(attention_scores, k=top_k, dim=1)  # [num_heads, top_k]
#         del attention_scores
        
#         indices_expanded = indices.unsqueeze(-1).expand(-1, -1, head_dim)  # [num_heads, top_k, head_dim]
#         top_image_keys = torch.gather(image_keys, 1, indices_expanded)  # [num_heads, top_k, head_dim]
#         top_image_values = torch.gather(image_values, 1, indices_expanded)  # [num_heads, top_k, head_dim]

#         mask = torch.zeros(num_heads, num_image_tokens, dtype=torch.bool, device=device)
#         mask.scatter_(1, indices, True)  # [num_heads, num_image_tokens]

#         remaining_mask = ~mask  # [num_heads, num_image_tokens]
#         remaining_count = num_image_tokens - top_k

#         cpu_key_states = image_keys[remaining_mask].view(num_heads, remaining_count, head_dim)
#         cpu_value_states = image_values[remaining_mask].view(num_heads, remaining_count, head_dim)

#         text_mask = ~image_mask
#         text_keys = key_states[0, :, text_mask]
#         text_values = value_states[0, :, text_mask]
#         gpu_key_states = torch.cat([text_keys, top_image_keys], dim=1).unsqueeze(0)
#         gpu_value_states = torch.cat([text_values, top_image_values], dim=1).unsqueeze(0)

#         return (
#             gpu_key_states,  # [1, num_heads, top_k, head_dim]
#             gpu_value_states,  # [1, num_heads, top_k, head_dim]
#             cpu_key_states.unsqueeze(0),  # [1, num_heads, remaining_count, head_dim]
#             cpu_value_states.unsqueeze(0),  # [1, num_heads, remaining_count, head_dim]
#         )


class FuseAttention(nn.Module):
    def __init__(self, model_config, dtype=torch.bfloat16):
        super().__init__()
        self.register_buffer(
            "alpha",
            torch.tensor(model_config.alpha, dtype=dtype)
        )

    def forward(self, A_gpu, A_cpu):
        return self.alpha * A_gpu + (1. - self.alpha) * A_cpu


class RandomProjectionAttention(nn.Module):
    def __init__(self, model_config, dtype=torch.bfloat16):
        super().__init__()
        self.head_dim = model_config.head_dim
        self.proj_dim = model_config.proj_dim
        self.R = torch.randn(self.head_dim, self.proj_dim, dtype=dtype) / (self.proj_dim ** 0.5)

    def prepare(self, K, V):
        self.K_proj = K @ self.R  # [bsz, num_heads, kv_len, proj_dim]
        self.V = V  # [bsz, num_heads, kv_len, head_dim]

    def forward(self, Q):
        # Q.shape [bsz, num_heads, q_len, head_dim]
        Q_proj = Q @ self.R  # [bsz, num_heads, q_len, proj_dim]
        scores = Q_proj @ self.K_proj.transpose(2, 3) / (self.proj_dim ** 0.5)  # [bsz, num_heads, q_len, kv_len]
        weights = F.softmax(scores, dim=-1)
        return weights @ self.V  # [bsz, num_heads, q_len, head_dim]


class KernelizedAttention(nn.Module):
    def __init__(self, model_config, dtype=torch.bfloat16, eps=1e-6):
        super().__init__()
        self.head_dim = model_config.head_dim
        self.dtype = dtype
        self.eps = eps

        self.KV = torch.tensor([], dtype=self.dtype)
        self.Z = torch.tensor([], dtype=self.dtype)

    @torch.jit.export
    def prepare(self, K, V):
        bsz, num_heads, kv_len, head_dim = K.shape
        K_reshaped = K.reshape(bsz * num_heads, kv_len, head_dim)
        V_reshaped = V.reshape(bsz * num_heads, kv_len, head_dim)

        K_phi = F.elu(K_reshaped) + 1
        K_phi = K_phi.transpose(1, 2)
        self.KV = torch.bmm(K_phi, V_reshaped)  # [bsz*num_heads, head_dim, head_dim]
        ones = torch.ones(bsz * num_heads, kv_len, 1, device='cpu', dtype=K.dtype)
        self.Z = torch.bmm(K_phi, ones)  # [bsz*num_heads, head_dim, 1]

    def forward(self, Q):
        bsz, num_heads, q_len, head_dim = Q.shape

        Q_reshaped = Q.reshape(bsz * num_heads, q_len, head_dim)
        Q_phi = F.elu(Q_reshaped) + 1
        
        numerator = torch.bmm(Q_phi, self.KV)  # [bsz*num_heads, q_len, head_dim]
        denominator = torch.bmm(Q_phi, self.Z) + self.eps  # [bsz*num_heads, q_len, 1]
        
        output = numerator / denominator
        output = output.reshape(bsz, num_heads, q_len, head_dim)
        return output


class SVDLowrankAttention(nn.Module):
    def __init__(self, model_config, dtype=torch.bfloat16):
        super().__init__()
        self.head_dim = model_config.head_dim
        self.proj_dim = model_config.proj_dim
        self.dtype = dtype

    def prepare(self, K, V):
        bsz, num_heads, kv_len, head_dim = K.shape

        K_reshaped = K.reshape(bsz * num_heads, kv_len, head_dim)
        self.V_reshaped = V.reshape(bsz * num_heads, kv_len, head_dim)

        K_mean = K_reshaped.mean(dim=1, keepdim=True)
        K_centered = K_reshaped - K_mean  # [bsz*num_heads, kv_len, head_dim]
        K_centered = K_centered.to(torch.float32)
        _, _, Vh = torch.linalg.svd(K_centered, full_matrices=False)
        Vh = Vh.to(self.dtype)
        self.B = Vh[..., :self.proj_dim, :].transpose(-2, -1)  # [bsz*num_heads, head_dim, proj_dim]
        
        self.K_proj = torch.bmm(K_reshaped, self.B)  # [bsz*num_heads, kv_len, proj_dim]

    def forward(self, Q):
        # Q.shape [bsz, num_heads, q_len, head_dim]
        bsz, num_heads, q_len, head_dim = Q.shape

        Q_reshaped = Q.reshape(bsz * num_heads, q_len, head_dim)
        Q_proj = torch.bmm(Q_reshaped, self.B)  # [bsz*num_heads, q_len, proj_dim]

        scores = torch.bmm(Q_proj, self.K_proj.transpose(1, 2)) / (self.proj_dim ** 0.5)  # [bsz*num_heads, q_len, kv_len]
        weights = F.softmax(scores, dim=-1)  # [bsz*num_heads, q_len, kv_len]

        output = torch.bmm(weights, self.V_reshaped)  # [bsz*num_heads, q_len, head_dim]

        output = output.reshape(bsz, num_heads, q_len, head_dim)
        return output


class LSHAttention(torch.nn.Module):
    def __init__(self, model_config, dtype=torch.bfloat16, eps=1e-8):
        super().__init__()
        self.bucket_size = model_config.bucket_size
        self.n_hashes = model_config.n_hashes
        self.dtype = dtype
        self.eps = eps

    def prepare(self, K, V):
        """
        K, V: [batch_size, num_head, seq_len, head_dim]
        """
        self.K = K
        self.V = V
        bsz, nhead, seqlen, d = K.shape

        self.K_norm = K / (torch.norm(K, dim=-1, keepdim=True) + 1e-8)  # [B, H, L, D]
        self.rotations = torch.randn(nhead, self.n_hashes, d, device=K.device, dtype=self.dtype)  # [H, n_hashes, D]

        # [B, H, L, D] @ [H, D, n_hashes] => [B, H, L, n_hashes]
        self.powers = 2 ** torch.arange(self.n_hashes, device=K.device)  # [n_hashes]
        K_hashes = torch.einsum('bhld,hnd->bhln', self.K_norm, self.rotations) > 0  # [B, H, L, n_hashes]
        self.K_buckets = K_hashes.long() @ self.powers  # [B, H, L]

        self.bucket2kidx = []
        for b in range(bsz):
            batch_list = []
            for h in range(nhead):
                buckets = self.K_buckets[b, h]  # [L]
                unique_buckets = torch.unique(buckets)
                bucket_map = {}
                for bucket in unique_buckets:
                    k_indices = (buckets == bucket).nonzero(as_tuple=True)[0]
                    bucket_map[int(bucket.item())] = k_indices
                batch_list.append(bucket_map)
            self.bucket2kidx.append(batch_list)

    def forward(self, Q):
        """
        Q: [batch_size, num_head, seq_len, head_dim]
        """
        bsz, nhead, seqlen, d = Q.shape
        Q_norm = Q / (torch.norm(Q, dim=-1, keepdim=True) + 1e-8)  # [B, H, L, D]

        Q_hashes = torch.einsum('bhld,hnd->bhln', Q_norm, self.rotations) > 0  # [B, H, L, n_hashes]
        Q_buckets = Q_hashes.long() @ self.powers  # [B, H, L]

        output = torch.zeros_like(Q)  # [B, H, L, D]

        for b in range(bsz):
            for h in range(nhead):
                q_buckets = Q_buckets[b, h]  # [L]
                k_buckets_map = self.bucket2kidx[b][h]
                unique_q_buckets = torch.unique(q_buckets)
                for bucket in unique_q_buckets:
                    q_indices = (q_buckets == bucket).nonzero(as_tuple=True)[0]
                    k_indices = k_buckets_map.get(int(bucket.item()), None)
                    if k_indices is None or len(k_indices) == 0 or len(q_indices) == 0:
                        continue
                    Q_bucket = Q[b, h, q_indices]      # [n_q, D]
                    K_bucket = self.K[b, h, k_indices] # [n_k, D]
                    V_bucket = self.V[b, h, k_indices] # [n_k, D]

                    scores = Q_bucket @ K_bucket.T / (d ** 0.5)  # [n_q, n_k]
                    scores_exp = torch.exp(scores - scores.max(dim=1, keepdim=True)[0])
                    weights = scores_exp / (scores_exp.sum(dim=1, keepdim=True) + self.eps)  # [n_q, n_k]

                    output[b, h, q_indices] += weights @ V_bucket  # [n_q, D]

        output = output / self.n_hashes
        return output


def make_cpu_attention(model_config):
    if model_config.cpu_attention_type == "random_projection_attention":
        model = RandomProjectionAttention(model_config)
    elif model_config.cpu_attention_type == "kernelized_attention":
        model = KernelizedAttention(model_config)
    elif model_config.cpu_attention_type == "svd_lowrank_attention":
        model = SVDLowrankAttention(model_config)
    elif model_config.cpu_attention_type == "lsh_attention":
        model = LSHAttention(model_config)
    else:
        raise NotImplementedError("unsupported cpu attention type {}".format(model_config.cpu_attention_type))
    
    model.eval()
    scripted_model = torch.jit.script(model)
    return scripted_model


if __name__ == "__main__":
    from time import time

    class ModelConfig:
        cpu_attention_type: str = "kernelized_attention"
        head_dim: int = 128
    model_config = ModelConfig()

    model = make_cpu_attention(model_config)

    begin_time = time()
    k = torch.rand(1, 28, 20000, 128)
    v = torch.rand(1, 28, 20000, 128)
    model.prepare(k, v)
    for _ in range(100):
        q = torch.rand(1, 28, 1, 128)
        model(q)
    end_time = time()
    print(end_time - begin_time)

    begin_time = time()
    for epoch in range(4):
        k = torch.rand(1, 28, 20000, 128)
        v = torch.rand(1, 28, 20000, 128)
        model.prepare(k, v)
        for _ in range(100):
            q = torch.rand(1, 28, 1, 128)
            model(q)
        end_time = time()
    print((end_time - begin_time) / 4.)
