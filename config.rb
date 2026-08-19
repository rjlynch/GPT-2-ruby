class Config < Struct.new(
  :vocab_size,
  :context_length,
  :emb_dim,
  :n_heads,
  :n_layers,
  :drop_rate,
  :qkv_bias,
  keyword_init: true
); end

GPT_CONFIG_124M = Config.new(
  emb_dim: 768,
  n_layers: 12,
  n_heads: 12,
  vocab_size: 50257,
  context_length: 1024,
  drop_rate: 0,
  qkv_bias: false
)

GPT_2_SMALL_124M_CONFIG = Config.new(
  emb_dim: 768,
  n_layers: 12,
  n_heads: 12,
  vocab_size: 50257,
  context_length: 1024,
  drop_rate: 0,
  qkv_bias: true
)
