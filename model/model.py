"""
File based on the original NanoGPT codebase.
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""
import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from model.embeddings import *

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-6, bias=False):
        """
        Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps: epsilon value
        :param bias: whether use bias term for RMSNorm
        """
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias
        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)
        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)
    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)
            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size
        rms_x = norm_x * d_x ** (-0.5)
        x_normed = x / (rms_x + self.eps)
        if self.bias:
            return self.scale * x_normed + self.offset
        return self.scale * x_normed

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        # Calculate survival probability
        keep_prob = 1 - self.drop_prob
        
        # Create a random binary mask for each sample in the batch
        # Shape: (batch_size, 1, 1, ...) to broadcast across all other dimensions
        shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize: becomes 0 or 1
        
        # Invert the mask and scale to maintain expected value
        output = x.div(keep_prob) * random_tensor
        return output

class GatedGELUFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
                                           
        if config.n_embd < 1024:
            self.expand_coef = 4 * config.n_embd
        else:
            self.expand_coef = 2 * config.n_embd
        self.wi_gate = nn.Linear(config.n_embd, self.expand_coef, bias=True)
        self.wi_act = nn.Linear(config.n_embd, self.expand_coef, bias=True)
        self.wo = nn.Linear(self.expand_coef, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        gate = self.wi_gate(x) # (batch, seq, d_ff)
        act_input = self.wi_act(x) # (batch, seq, d_ff)
        act_output = F.gelu(act_input) # e.g. new_gelu or standard gelu
        hidden = gate * act_output # elementwise mul
        out = self.wo(hidden)
        return out
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.expand_coef = 4
        self.c_fc = nn.Linear(config.n_embd, self.expand_coef * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(self.expand_coef * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 96
    n_layer: int = 16
    n_head: int = 16
    n_embd: int = 1024
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears/LayerNorms
    use_flash: bool = True
    ff_proj: str = 'mlp' # 'relu', 'gelu', 'gated-gelu' etc.
    normalization_layer: str = 'layernorm' # 'layernorm', 'rmsnorm'
    layer_norm_position: str = 'pre' # 'pre', 'post', 'pre_post'
    layer_norm_epsilon: float = 1e-5
    # ---------- positional embedding key -----------
    positional_embedding: str = "learned" # "positional", "sinusoidal", "scaledsinosoidal", "learned", "learned-rand", "random-noise", "abacus", "pc"
    fire_max_length: int = 0
                                           
    use_deepnorm: bool = True
    clamp_fp16_activations: bool = True
    T_backward: float = 1.0
    T_forward: float = 1.0
    abacus_maxK: int = 100
    drop_path_rate: float = 0.02 # stochastic depth rate, default 0.0
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, rel_pos_embed=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // self.n_head
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.use_flash = config.use_flash
        # causal mask
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        # rel_pos_embed, e.g. RPE/FIRE/Rotary
        self.rel_pos_embed = rel_pos_embed
    def forward(self, x):
                                            
        if isinstance(self.rel_pos_embed, RPE):
            y = self.rel_pos_embed(x) # shape (B, T, C)
            y = self.resid_dropout(self.c_proj(y))
            return y
        B, T, C = x.shape
        # Q,K,V
        qkv = self.c_attn(x) # (B,T,3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                  
        fire_bias = None
        alibi_bias = None
        t5_bias = None
        kerple_bias = None
        rpe_bias = None
        if isinstance(self.rel_pos_embed, RPEBias):
            rpe_bias = self.rel_pos_embed(q)
        if isinstance(self.rel_pos_embed, FIRE):
            fire_bias = self.rel_pos_embed(T, x.device)
        elif isinstance(self.rel_pos_embed, Rotary):
            q, k = self.rel_pos_embed(q, k)
        elif isinstance(self.rel_pos_embed, AlibiPositionalBias):
            alibi_bias = self.rel_pos_embed(T, x.device)
        elif isinstance(self.rel_pos_embed, T5RelativePositionBias):
            t5_bias = self.rel_pos_embed(T, x.device)
        elif isinstance(self.rel_pos_embed, KerpleRelativeBias):
            kerple_bias = self.rel_pos_embed(T, x.device)
        # scaled dot product
        if rpe_bias is not None:
            att = ((q @ k.transpose(-2,-1)) + rpe_bias)/ math.sqrt(self.head_dim)
        else:
            att = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        if fire_bias is not None:
            att = att + fire_bias
        if alibi_bias is not None:
            att = att + alibi_bias
        if t5_bias is not None:
            att = att + t5_bias
        if kerple_bias is not None:
            att = att + kerple_bias
        # causal mask
        att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class Block(nn.Module):
    def __init__(self, config: GPTConfig, rel_pos_embed=None):
        super().__init__()
        # --------------- DeepNorm (alpha) ---------------
        if config.use_deepnorm:
                                                          
            self.alpha = math.pow(2.0 * config.n_layer, 0.25)
        else:
            self.alpha = 1.0
                                             
        if config.normalization_layer == 'rmsnorm':
            self.norm1 = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
            self.norm2 = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
            if config.layer_norm_position == 'pre_post':
                self.norm1_2 = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
                self.norm2_2 = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        elif config.normalization_layer == 'layernorm':
            self.norm1 = LayerNorm(config.n_embd, bias=config.bias)
            self.norm2 = LayerNorm(config.n_embd, bias=config.bias)
            if config.layer_norm_position == 'pre_post':
                self.norm1_2 = LayerNorm(config.n_embd, bias=config.bias)
                self.norm2_2 = LayerNorm(config.n_embd, bias=config.bias)
        else:
            raise ValueError("Unknown normalization layer")
        self.attn = CausalSelfAttention(config, rel_pos_embed=rel_pos_embed)
        if config.ff_proj == 'gated-gelu':
            self.ffn = GatedGELUFeedForward(config)
        else:
            self.ffn = MLP(config)
        self.layer_norm_position = config.layer_norm_position
        self.config = config              
        self.drop_path_rate = config.drop_path_rate
        self.drop_path = DropPath(config.drop_path_rate) if config.drop_path_rate > 0. else nn.Identity()
        
    def forward(self, x):
                                  
        # Store original input for debugging in except block if needed
        original_x = x
                                  
        try:
            if self.layer_norm_position in ['pre', 'pre_post']:
                x_normed = self.norm1(x)
            else:
                x_normed = x
            attn_output = self.attn(x_normed)
            x = self.alpha *original_x +self.drop_path(attn_output)

            if self.config.clamp_fp16_activations and x.dtype == torch.float16:
                clamp_val = torch.where(
                    torch.isinf(x).any(),
                    torch.finfo(x.dtype).max - 1000,
                    torch.finfo(x.dtype).max,
                )
                x = torch.clamp(x, min=-clamp_val, max=clamp_val)
            if self.layer_norm_position == 'post':
                x = self.norm1(x)
            elif self.layer_norm_position == 'pre_post':
                x = self.norm1_2(x)
        except RuntimeError as e:
            print(f"\n{'='*20} CATCHING RUNTIME ERROR INSIDE ATTENTION BLOCK {'='*20}")
            print(f"ERROR type: {type(e)}")
            print(f"ERROR message: {e}")
            print(f"--- Debug Info (Attention Block) ---")
            # Print info about tensors available just before the error
            if 'x_normed' in locals():
                print(f" x_normed shape: {x_normed.shape}, dtype: {x_normed.dtype}, has NaN: {torch.isnan(x_normed).any().item()}, has Inf: {torch.isinf(x_normed).any().item()}")
            else:
                print(" x_normed not computed yet.")
            if 'attn_output' in locals():
                print(f" attn_output shape: {attn_output.shape}, dtype: {attn_output.dtype}, has NaN: {torch.isnan(attn_output).any().item()}, has Inf: {torch.isinf(attn_output).any().item()}")
            else:
                print(" attn_output not computed yet (error might be during its calculation).")
            print(f" original_x shape: {original_x.shape}, dtype: {original_x.dtype}, has NaN: {torch.isnan(original_x).any().item()}, has Inf: {torch.isinf(original_x).any().item()}")
            print(f" self.alpha value: {self.alpha}")
            print(f"--- End Debug Info (Attention Block) ---")
            # Re-raise the exception to stop execution or allow outer try-except to catch
            raise e
                                
        if self.layer_norm_position in ['pre', 'pre_post']:
            x_normed = self.norm2(x)
        else:
            x_normed = x
        ffn_out = self.ffn(x_normed)
                     
        x = self.alpha * x+ self.drop_path(ffn_out)
        if self.config.clamp_fp16_activations and x.dtype == torch.float16:
            clamp_val = torch.where(
                torch.isinf(x).any(),
                torch.finfo(x.dtype).max - 1000,
                torch.finfo(x.dtype).max,
            )
            x = torch.clamp(x, min=-clamp_val, max=clamp_val)
        if self.layer_norm_position == 'post':
            x = self.norm2(x)
        elif self.layer_norm_position == 'pre_post':
            x = self.norm2_2(x)
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.T_backward = config.T_backward
        self.T_forward = config.T_forward
        # (1) build absolute pos embed & rel pos embed
        abs_embed, rel_embed = self._build_position_embedding(config)
        # create main modules
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd), # token embed
            "abs_embed": abs_embed,
            "rpe_embed": rel_embed,
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config, rel_pos_embed=rel_embed)
                                for _ in range(config.n_layer)]),
            "ln_f": LayerNorm(config.n_embd, bias=config.bias),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying
        self.transformer["wte"].weight = self.lm_head.weight
        # init weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0,
                                      std=0.02/math.sqrt(2 * config.n_layer))
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def _build_position_embedding(self, config: GPTConfig):
        abs_embed = None
        rel_embed = None
        pe = config.positional_embedding.lower()
        # absolute embeddings
        if pe == "positional":
            abs_embed = PositionalEmbedding(demb=config.n_embd)
        elif pe == "sinusoidal":
            abs_embed = SinusoidalPositional(embedding_dim=config.n_embd,
                                             max_seq_length=config.block_size)
        elif pe == "scaledsinosoidal":
            abs_embed = ScaledSinosoidal(embedding_dim=config.n_embd,
                                         max_seq_length=config.block_size)
        elif pe == "learned":
            abs_embed = LearnablePositional(embedding_dim=config.n_embd,
                                            max_seq_length=config.block_size)
        elif pe == "learned-rand":
            abs_embed = LearnablePositionalRand(embedding_dim=config.n_embd,
                                                max_seq_length=config.block_size)
        elif pe == "random-noise":
            abs_embed = RandomNoise(embedding_dim=config.n_embd,
                                    max_seq_length=config.block_size)
        elif pe == "abacus":
            abs_embed = Abacus(digit_tokens=[1, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
                               embedding_dim=config.n_embd,
                               max_seq_length=config.block_size,
                               max_k=config.abacus_maxK)
        elif pe == "pc":
            abs_embed = PositionCoupling(digit_tokens=[1, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
                                         embedding_dim=config.n_embd,
                                         max_seq_length=config.block_size,
                                         max_k=99)
        # relative embeddings
        if pe == "rpe":
            rel_embed = RPE(d_model=config.n_embd,
                            num_heads=config.n_head,
                            max_len=config.block_size,
                            dropout=config.dropout)
        elif pe == "rpebias":
            rel_embed = RPEBias(d_model=config.n_embd,
                              num_heads=config.n_head,
                              max_len=config.block_size,
                              dropout=config.dropout)
        elif pe == "rope":
            rel_embed = Rotary(dim=(config.n_embd // config.n_head))
        elif pe == "fire":
            rel_embed = FIRE(num_heads=config.n_head,
                             max_length=config.fire_max_length)
        elif pe == "alibi":
            rel_embed = AlibiPositionalBias(num_heads=config.n_head,
                                            max_sequence_length=config.block_size)
            abs_embed = None
        elif pe == "t5-rpe":
            rel_embed = T5RelativePositionBias(num_buckets=32, max_distance=128,
                                             n_heads=config.n_head)
            abs_embed = None
        elif pe == "kerple":
            rel_embed = KerpleRelativeBias(
                num_heads=config.n_head,
                max_seq_len=config.block_size,
                kernel_type="log", # or "power"
                learnable=True
            )
        return abs_embed, rel_embed
    
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None, scaler_factor=1.0):
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block_size is {self.config.block_size}"
        )
        # token embeddings
        tok_emb = self.transformer["wte"](idx) # (B,T,n_embd)
        # absolute embeddings
        if self.transformer.abs_embed is not None:
            pos_emb = self.transformer.abs_embed(idx)
            x = tok_emb + pos_emb
        else:
            x = tok_emb
        x = self.transformer["drop"](x)
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)
        if targets is not None:
            logits = self.lm_head(x) # (B,T,vocab)
        else:
            logits = self.lm_head(x[:, [-1], :]) # (B,1,vocab)
        # loss / logits
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), # [B*T, vocab]
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            loss = None
        return logits, loss
    
    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.abs_embed.embedding.weight = nn.Parameter(self.transformer.abs_embed.embedding.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
    
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        config_args['positional_embedding'] = "learned"
        config_args['use_flash'] = True
        config_args['clamp_fp16_activations'] = False
        config_args['use_deepnorm'] = False
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        config = GPTConfig(**config_args)
        model = GPT(config)
        print(config_args)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'ffn.c_fc.weight', 'ffn.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
                                
            mapped_key = k
            if "ln_1" in mapped_key:
                mapped_key = mapped_key.replace("ln_1", "norm1")
            if "ln_2" in mapped_key:
                mapped_key = mapped_key.replace("ln_2", "norm2")
            if "mlp" in mapped_key:
                mapped_key = mapped_key.replace("mlp", "ffn")
            if "transformer.wpe.weight" in mapped_key:
                mapped_key = mapped_key.replace("transformer.wpe.weight", "transformer.abs_embed.embedding.weight")
                                    
            if mapped_key in sd:
                if any(mapped_key.endswith(w) for w in transposed):
                                        
                    assert sd_hf[k].shape[::-1] == sd[mapped_key].shape
                    with torch.no_grad():
                        sd[mapped_key].copy_(sd_hf[k].t())
                else:
                    assert sd_hf[k].shape == sd[mapped_key].shape, f"mismatched shapes: {k}: {sd_hf[k].shape} != {sd[mapped_key].shape}"
                    with torch.no_grad():
                        sd[mapped_key].copy_(sd_hf[k])
            else:
                print(f"Skipping key {k} (mapped to {mapped_key}) not found in target model")
        return model
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay = set()
        no_decay = set()
        for full_name, param in self.named_parameters():
            if full_name.endswith(".bias"):
                no_decay.add(full_name)
            elif full_name.endswith(".weight"):
                if "ln" in full_name or "embedding" in full_name:
                    no_decay.add(full_name)
                else:
                    decay.add(full_name)
            else:
                no_decay.add(full_name)
        print("weight decay params:", decay)
        print("no decay params:", no_decay)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx