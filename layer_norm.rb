class LayerNorm < Torch::NN::Module
  def initialize(emb_dim)
    super()
    @eps = 1e-5
    @scale = Torch::NN::Parameter.new(Torch.ones(emb_dim))
    @shift = Torch::NN::Parameter.new(Torch.zeros(emb_dim))
  end

  def forward(x)
    mean = x.mean(dim: -1, keepdim: true)
    var = x.var(dim: -1, keepdim: true, unbiased: false)
    norm_x = (x - mean) / Torch.sqrt(var + @eps)

    @scale * norm_x + @shift
  end
end
