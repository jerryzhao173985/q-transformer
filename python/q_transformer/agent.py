134|        text_embed_shape: int | tuple[int, ...],
135|        device: torch.device | None = None  # <-- new optional parameter
136|    ):
137|        super().__init__()
138|        self.state_shape = state_shape
139|        self.text_embed_shape = cast_tuple(text_embed_shape)
140|        # if no device provided, use MPS if available, otherwise cpu
141|        if device is None:
142|            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
143|        self.register_buffer('dummy', torch.zeros(0, device=device), persistent = False) 