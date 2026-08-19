class Gelu < Torch::NN::Module
  def initialize()
    super()
  end

  def forward(x)
    0.5 * x * (
      1 + Torch.tanh(
       Torch.sqrt(Torch.tensor(2.0 / Math::PI)) * (x + 0.044715 * x**3)
      )
    )
  end
end
