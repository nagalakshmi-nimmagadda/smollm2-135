from dataclasses import dataclass
from typing import Optional

@dataclass
class SmolLM2Config:
    hidden_size: int = 576
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    intermediate_size: int = 1536
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.041666666666666664
    rms_norm_eps: float = 1.0e-5
    vocab_size: int = 49152
    rope_theta: float = 10000.0
    rope_interleaved: bool = False
    rope_scaling: Optional[dict] = None
    tie_word_embeddings: bool = True
    bos_token_id: int = 0
    eos_token_id: int = 0
    pad_token_id: None = None
    is_llama_config: bool = True
    pretraining_tp: int = 1
    use_cache: bool = True
    
    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.kv_head_dim = self.hidden_size // self.num_key_value_heads 