require "safetensors"

# curl -fL \
#   -o weights/gpt2-small-124M.safetensors \
#   https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/gpt2-small-124M.safetensors

module GPT2Weights
  extend self

  # Map python invar names to the ones I've used
  def ruby_key(key)
    key
      .gsub(".att.W_query.", ".att.w_query.")
      .gsub(".att.W_key.", ".att.w_key.")
      .gsub(".att.W_value.", ".att.w_value.")
      .gsub(".norm1.", ".norm_1.")
      .gsub(".norm2.", ".norm_2.")
  end

  def weights
    Safetensors::Torch
      .load_file("weights/gpt2-small-124M.safetensors")
      .to_h { |key, tensor| [ruby_key(key), tensor] }
  end
end
