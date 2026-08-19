require "tokenizers"
require "torch"
require "safetensors"
require_relative "config"
require_relative "gpt_model"
require_relative "layer_norm"
require_relative "gelu"
require_relative "feed_forward"
require_relative "multi_head_attention"
require_relative "transformer_block"
require_relative "text_helpers"
require_relative "gpt_2_weights"

tokenizer = Tokenizers.from_pretrained("gpt2")

device = Torch.device("cpu")

weights = GPT2Weights.weights

gpt = GPTModel.new(GPT_2_SMALL_124M_CONFIG)
gpt.to(device)
gpt.load_state_dict(weights)
gpt.eval

while true do
  print "> "
  input = gets.chomp

  token_ids = TextHelpers.generate(
    model: gpt,
    idx: TextHelpers.text_to_token_ids(input, tokenizer).to(device),
    max_new_tokens: 50,
    context_size: GPT_2_SMALL_124M_CONFIG.context_length,
    topk: 50,
    temperature: 1.5
  )

  output = TextHelpers.token_ids_to_text(token_ids, tokenizer)

  puts
  puts output
  puts
end
