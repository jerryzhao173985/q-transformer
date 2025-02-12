from __future__ import annotations

from random import randrange

import torch
from torch.utils.data import Dataset

from q_transformer.tensor_typing import Float, Int, Bool
from q_transformer.agent import BaseEnvironment

class MockEnvironment(BaseEnvironment):
    def __init__(
        self,
        *,
        state_shape: tuple[int, ...] = (3, 6, 224, 224),
        text_embed_shape: int | tuple[int, ...] = 512,
        device: torch.device | None = None
    ):
        # Always try to use MPS by default
        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            
        super().__init__(
            state_shape=state_shape,
            text_embed_shape=text_embed_shape,
            device=device
        )

    def init(self) -> tuple[
        str | None,
        Float['...']
    ]:
        return 'please clean the kitchen', torch.randn(
            self.state_shape, 
            device=self.device,
            dtype=torch.float32  # Explicit dtype for MPS
        )

    def forward(self, actions) -> tuple[
        Float[''],
        Float['...'],
        Bool['']
    ]:
        rewards = torch.randn((), device=self.device, dtype=torch.float32)
        next_states = torch.randn(self.state_shape, device=self.device, dtype=torch.float32)
        done = torch.zeros((), device=self.device, dtype=torch.bool)

        return rewards, next_states, done

class MockReplayDataset(Dataset):
    def __init__(
        self,
        length = 10000,
        num_actions = 1,
        num_action_bins = 256,
        video_shape = (6, 224, 224),
        device = None
    ):
        self.length = length
        self.num_actions = num_actions
        self.num_action_bins = num_action_bins
        self.video_shape = video_shape
        self.device = get_device(device)

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        device = self.device
        instruction = "please clean the kitchen"
        state = torch.randn(3, *self.video_shape, device=device)

        if self.num_actions == 1:
            action = torch.tensor(randrange(self.num_action_bins + 1), device=device)
        else:
            action = torch.randint(0, self.num_action_bins + 1, (self.num_actions,), device=device)

        next_state = torch.randn(3, *self.video_shape, device=device)
        reward = torch.tensor(randrange(2), device=device)
        done = torch.tensor(randrange(2), dtype=torch.bool, device=device)

        return instruction, state, action, next_state, reward, done

class MockReplayNStepDataset(Dataset):
    def __init__(
        self,
        length = 10000,
        num_steps = 2,
        num_actions = 1,
        num_action_bins = 256,
        video_shape = (6, 224, 224),
        device = None
    ):
        self.num_steps = num_steps
        self.time_shape = (num_steps,)
        self.length = length
        self.num_actions = num_actions
        self.num_action_bins = num_action_bins
        self.video_shape = video_shape
        self.device = get_device(device)

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        device = self.device
        action_dims = (self.num_actions,) if self.num_actions > 1 else tuple()

        instruction = "please clean the kitchen"
        state = torch.randn(*self.time_shape, 3, *self.video_shape, device=device)
        action = torch.randint(0, self.num_action_bins + 1, (*self.time_shape, *action_dims), device=device)
        next_state = torch.randn(3, *self.video_shape, device=device)
        reward = torch.randint(0, 2, self.time_shape, device=device)
        done = torch.zeros(self.time_shape, dtype=torch.bool, device=device)

        return instruction, state, action, next_state, reward, done
