class GPTModel < Torch::NN::Module
  def initialize(cfg)
    super()

    @tok_emb = Torch::NN::Embedding.new(cfg.vocab_size, cfg.emb_dim)

    @pos_emb = Torch::NN::Embedding.new(cfg.context_length, cfg.emb_dim)

    @drop_emb = Torch::NN::Dropout.new(p: cfg.drop_rate)

    @trf_blocks = Torch::NN::Sequential.new(
      *cfg.n_layers.times.map { TransformerBlock.new(cfg) }
    )

    @final_norm = LayerNorm.new(cfg.emb_dim)

    @out_head = Torch::NN::Linear.new(cfg.emb_dim, cfg.vocab_size, bias: false)
  end

  def forward(in_idx)
    batch_size, seq_len, _embdim = in_idx.shape
    tok_embeds = @tok_emb.forward(in_idx)
    pos_embeds = @pos_emb.forward(Torch.arange(seq_len, device: in_idx.device))

    x = tok_embeds + pos_embeds
    x = @drop_emb.forward(x)
    x = @trf_blocks.forward(x)
    x = @final_norm.forward(x)
    logits = @out_head.forward(x)

    logits
  end
end
