class TransformerBlock < Torch::NN::Module
  def initialize(cfg)
    super()

    @att = MultiHeadAttention.new(
      cfg.emb_dim,
      cfg.emb_dim,
      cfg.context_length,
      cfg.drop_rate,
      cfg.n_heads,
      qkv_bias: cfg.qkv_bias
    )

    @ff = FeedForward.new(cfg)
    @norm_1 = LayerNorm.new(cfg.emb_dim)
    @norm_2 = LayerNorm.new(cfg.emb_dim)
    @drop_shortcut = Torch::NN::Dropout.new(p: cfg.drop_rate)
  end

  def forward(x)
    shortcut = x
    x = @norm_1.forward(x)
    x = @att.forward(x)
    x = @drop_shortcut.forward(x)
    x = x + shortcut

    shortcut = x
    x = @norm_2.forward(x)
    x = @ff.forward(x)

    x = @drop_shortcut.forward(x)
    x = x + shortcut

    x
  end
end
