class RotaryEmbedding(Module):
    def __init__(self, dim, omega = 10000):
        super().__init__()
        inv_freq = 1.0 / (omega ** (torch.arange(0, dim, 4).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, height_width):
        device, dtype = self.inv_freq.device, self.inv_freq.dtype
        axial_pos = torch.arange(height_width, device = device).type(dtype)
        freqs = torch.einsum('i, j -> i j', axial_pos, self.inv_freq)
        freqs = repeat(freqs, '... f -> ... (f c)', c = 2)
        freqs = torch.broadcast_tensors(freqs[None, :, :], freqs[:, None, :])
        freqs = torch.cat(freqs, dim = -1)
        return rearrange(freqs, '... f -> (...) f')

def rotate_half(x):
    x1, x2 = rearrange(x, '... (d c) -> ... d c', c = 2).unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d c -> ... (d c)')

def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin() 