module TextHelpers
  extend self

  def generate_text_simple(model:, idx:, max_new_tokens:, context_size:)
    max_new_tokens.times do
      idx_cond = idx[0.., -context_size..]

      Torch.no_grad do
        logits = model.forward(idx_cond)
        logits = logits[0.., -1, 0..]
        probas = Torch.softmax(logits, dim: -1)
        idx_next = Torch.argmax(probas, dim: -1, keepdim: true)
        idx = Torch.cat([idx, idx_next], dim: 1)

      end
    end
    idx
  end

  def generate(model:, idx:, max_new_tokens:, context_size:, temperature: 0.0, topk: nil, eos_id: nil)
    max_new_tokens.times do
      idx_cond = idx[0.., -context_size..]

      Torch.no_grad do
        all_next_token_logits = model.forward(idx_cond)
        logits = all_next_token_logits[0.., -1, 0..]

        if !topk.nil?
          top_logits, _top_pos = Torch.topk(logits, topk)
          min_val = top_logits[0.., -1]
          condition = Torch.lt(logits, min_val)
          logits = Torch.where(
            condition: condition,
            input: Torch.tensor(-Float::INFINITY).to(logits.device),
            other: logits
          )
        end

        if temperature > 0.0
          logits = logits / temperature
          probas = Torch.softmax(logits, dim: -1)
          idx_next = Torch.multinomial(probas, num_samples: 1)
        else
          probas = Torch.softmax(logits, dim: -1)
          idx_next = Torch.argmax(probas, dim: -1, keepdim: true)
        end


        break if idx_next == eos_id

        idx = Torch.cat([idx, idx_next], dim: 1)
      end
    end

    idx
  end

  def text_to_token_ids(text, tokenizer)
    encoded = tokenizer.encode(text, add_special_tokens: ['<endoftext|>']).ids
    encoded_tensor = Torch.tensor(encoded).unsqueeze(0)
    encoded_tensor
  end

  def token_ids_to_text(token_ids, tokenizer)
    flat = token_ids.squeeze(0)
    tokenizer.decode(flat.to_a).to_s
  end
end
