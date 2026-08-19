class MultiHeadAttention < Torch::NN::Module
  def initialize(d_in, d_out, context_length, dropout, num_heads, qkv_bias: false)
    super()

    raise "d_out must be divisible by num_heads" unless d_out % num_heads == 0

    @d_out = d_out
    @num_heads = num_heads
    @head_dim = d_out / num_heads
    @w_query = Torch::NN::Linear.new(d_in, d_out, bias: qkv_bias)
    @w_key = Torch::NN::Linear.new(d_in, d_out, bias: qkv_bias)
    @w_value = Torch::NN::Linear.new(d_in, d_out, bias: qkv_bias)
    @out_proj = Torch::NN::Linear.new(d_out, d_out)
    @dropout = Torch::NN::Dropout.new(p: dropout)
    register_buffer(
      "mask",
      Torch.triu(Torch.ones(context_length, context_length), diagonal: 1)
    )
  end

  def forward(x)
    b, num_tokens, _d_in = x.shape
    keys = @w_key.forward(x)
    queries = @w_query.forward(x)
    values = @w_value.forward(x)

    keys = keys.view(b, num_tokens, @num_heads, @head_dim)
    values = values.view(b, num_tokens, @num_heads, @head_dim)
    queries = queries.view(b, num_tokens, @num_heads, @head_dim)

    keys = keys.transpose(1, 2)
    queries = queries.transpose(1, 2)
    values = values.transpose(1, 2)

    attn_scores = queries.matmul(keys.transpose(2, 3))

    mask_bool = @mask.bool[0...num_tokens, 0...num_tokens]

    attn_scores = attn_scores.masked_fill(mask_bool, -1e4)

    attn_weights = Torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim: -1)

    attn_weights = @dropout.forward(attn_weights)

    context_vec = (attn_weights.matmul(values)).transpose(1, 2)

    context_vec = context_vec.contiguous.view(b, num_tokens, @d_out)

    context_vec = @out_proj.forward(context_vec)

    context_vec
  end
end
