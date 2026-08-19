class FeedForward < Torch::NN::Module
  def initialize(cfg)
    super()

    @layers = Torch::NN::Sequential.new(
      Torch::NN::Linear.new(cfg.emb_dim, 4 * cfg.emb_dim),
      Torch::NN::GELU.new(approximate: "tanh"),
      Torch::NN::Linear.new(4 * cfg.emb_dim, cfg.emb_dim)
    )
  end

  def forward(x)
    @layers.forward(x)
  end
end
