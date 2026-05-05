# models/steering.py — real residual-stream injection via PyTorch forward hooks
#
# How it works:
#   1. We register a hook on model.model.layers[STEERING_LAYER]
#   2. The hook adds the steering vector to the hidden states DURING the forward pass
#   3. The hook is removed immediately after generation so it doesn't leak

import torch
from config import STEERING_LAYER, STEERING_ALPHA


class SteeringHook:
    """
    Context manager that injects a steering vector into one transformer layer.

    Usage
    -----
    with SteeringHook(model, steering_vec):
        outputs = model.generate(...)
    """

    def __init__(self, model, steering_vector: torch.Tensor, alpha: float = STEERING_ALPHA):
        self.model          = model
        self.steering_vector = steering_vector   # shape: (hidden_dim,)
        self.alpha           = alpha
        self._handle         = None

    def _hook_fn(self, module, input, output):
        """
        output is typically a tuple: (hidden_states, ...).
        We modify hidden_states in place and return the modified tuple.
        """
        hidden = output[0]                         # (batch, seq_len, hidden_dim)

        vec = self.steering_vector.to(hidden.device, hidden.dtype)

        # Broadcast across batch and sequence dims
        # Scale by the mean norm of hidden states for stability
        scale = hidden.norm(dim=-1, keepdim=True).mean() / (vec.norm() + 1e-8)
        hidden = hidden + self.alpha * scale * vec

        # Return the modified tuple (preserve any extra outputs from the layer)
        return (hidden,) + output[1:]

    def __enter__(self):
        layer = self.model.model.layers[STEERING_LAYER]
        self._handle = layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *args):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    steering_vector: torch.Tensor,
    max_new_tokens: int = 300,
    alpha: float = STEERING_ALPHA,
) -> str:
    """
    Generate a response with the steering vector injected into the residual stream.

    Parameters
    ----------
    model           : loaded LLaMA model
    tokenizer       : matching tokenizer
    prompt          : the user's query (already formatted as needed)
    steering_vector : 1-D tensor of shape (hidden_dim,)
    max_new_tokens  : generation budget
    alpha           : injection strength multiplier

    Returns
    -------
    str : decoded model response (prompt stripped)
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(next(model.parameters()).device)

    input_len = inputs["input_ids"].shape[1]

    with SteeringHook(model, steering_vector, alpha=alpha):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

    # Strip the input prompt tokens; only return newly generated tokens
    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()