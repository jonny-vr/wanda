import torch.nn as nn

class ProxyCatcher(nn.Module):
    """
    Replaces the first decoder layer during calibration.
    Captures hidden-states and any kwargs (attention_mask,
    position_ids, position_embeddings, …) then aborts the
    forward pass by raising StopIteration.
    Attribute look-ups are transparently forwarded, so
    Qwen-3 can still read `attention_type`, etc.
    """
    def __init__(self, module, cache):
        super().__init__()
        self.module = module
        self.cache  = cache

    def forward(self, hidden_states, **kwargs):
        self.cache["inp"] = hidden_states.detach()
        self.cache.update(kwargs)          # save *all* kw-args
        raise StopIteration

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
