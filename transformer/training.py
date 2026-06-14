"""Loss and rule-error estimation."""
import random
import torch
from data import get_batch_from_sequences


@torch.no_grad()
def estimate_loss(model, train_sequences, val_sequences, block_size, batch_size, eval_iterations):
    """
    Average loss on 'train' and 'validation' splits.
    """
    out = {}
    model.eval()

    for split_name, sequences in [("train", train_sequences), ("validation", val_sequences)]:
        losses = torch.zeros(eval_iterations)
        for i in range(eval_iterations):
            X, Y = get_batch_from_sequences(sequences, block_size, batch_size)
            _, loss = model(X, Y)
            losses[i] = loss.item()
        out[split_name] = losses.mean()

    model.train()
    return out

@torch.no_grad()
def estimate_rule_error(model, generator, decode, block_size, num_samples=20, seq_length=30):
    """
    Generate sequences and check rule error.
    Returns the fraction of CONSTRAINED positions that violate the rule.
    Uses the generator's `valence_mask()` to determine which positions are constrained.
    """
    model.eval()
    
    total_constrained = 0
    incorrect_constrained = 0
    
    vocab_size = model.token_embedding.weight.shape[0]
    
    for _ in range(num_samples):
        # Generate a sequence
        start_token = random.randint(0, vocab_size - 1)
        start = torch.tensor([[start_token]], dtype=torch.long)
        sample = model.generate(start, max_new_tokens=seq_length - 1)[0].tolist()
        generated_integers = decode(sample)
        
        # Verify the sequence
        correctness, _ = generator.verify_sequence(generated_integers)
        valence = generator.valence_mask(generated_integers)
        for i, is_constrained in enumerate(valence):
            if i < len(correctness) and is_constrained:
                total_constrained += 1
                if correctness[i] == 0:
                    incorrect_constrained += 1
    
    model.train()
    return incorrect_constrained / total_constrained if total_constrained > 0 else 0.0
