import torch

from q_transformer import (
    QRoboticTransformer,
    QLearner,
    Agent,
    ReplayMemoryDataset
)

# the attention model

model = QRoboticTransformer(
    vit = dict(
        num_classes = 1000,
        dim_conv_stem = 64,
        dim = 64,
        dim_head = 64,
        depth = (2, 2, 5, 2),
        window_size = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1
    ),
    num_actions = 8,
    action_bins = 256,
    depth = 1,
    heads = 8,
    dim_head = 64,
    cond_drop_prob = 0.2,
    dueling = True
)

# you need to supply your own environment, by overriding BaseEnvironment

from q_transformer.mocks import MockEnvironment

env = MockEnvironment(
    state_shape = (3, 6, 224, 224),
    text_embed_shape = (768,)
)

# env.init()     should return instructions and initial state: Tuple[str, Tensor[*state_shape]]
# env(actions)   should return rewards, next state, and done flag: Tuple[Tensor[()], Tensor[*state_shape], Tensor[()]]

# agent is a class that allows the q-model to interact with the environment to generate a replay memory dataset for learning

agent = Agent(
    model,
    environment = env,
    num_episodes = 1000,
    max_num_steps_per_episode = 100,
)

agent()

# Q learning on the replay memory dataset on the model

q_learner = QLearner(
    model,
    dataset = ReplayMemoryDataset(),
    num_train_steps = 10000,
    learning_rate = 3e-4,
    batch_size = 4,
    grad_accum_every = 16,
    checkpoint_dir = './my_training_run',  # Specify checkpoint directory
    save_freq = 200  # Save every 1000 steps
)

# Train with automatic checkpointing
q_learner()

# after much learning
# your robot should be better at selecting optimal actions

video = torch.randn(2, 3, 6, 224, 224)

instructions = [
    'bring me that apple sitting on the table',
    'please pass the butter'
]

actions = model.get_optimal_actions(video, instructions)

# # Later, to load a trained model for inference:
# loaded_model = QRoboticTransformer.load_from_checkpoint(
#     'my_training_run/checkpoint_latest/checkpoint.pt',
#     vit=dict(...),  # Same configuration as original model
#     num_actions=8,
#     action_bins=256,
#     # ... other model parameters ...
# )

# # Use the loaded model
# actions = loaded_model.get_optimal_actions(video, instructions)