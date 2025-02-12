import torch
from pathlib import Path
from q_transformer import (
    QRoboticTransformer,
    QLearner,
    Agent,
    ReplayMemoryDataset
)
import wandb
from datetime import datetime

# Function to load latest checkpoint or create new model
def get_model(checkpoint_dir='./my_training_run', new_model=False):
    model = QRoboticTransformer(
        vit = dict(
            num_classes = 1000,
            dim_conv_stem = 128,
            dim = 128,
            dim_head = 64,
            depth = (2, 2, 5, 2),
            window_size = 7,
            mbconv_expansion_rate = 4,
            mbconv_shrinkage_rate = 0.25,
            dropout = 0.1
        ),
        num_actions = 8,
        action_bins = 256,
        depth = 2,
        heads = 8,
        dim_head = 64,
        cond_drop_prob = 0.2,
        dueling = True
    )
    
    if not new_model:
        checkpoint_path = Path(checkpoint_dir) / 'checkpoint_metadata.json'
        if checkpoint_path.parent.exists():
            try:
                loaded_model = QRoboticTransformer.load_from_checkpoint(
                    str(checkpoint_path.parent / 'latest' / 'checkpoint.pt'),
                    **model.config
                )
                print(f"Loaded model from {checkpoint_path}")
                return loaded_model
            except Exception as e:
                print(f"Could not load checkpoint, starting fresh: {str(e)}")
    
    return model

# Calculate optimal batch size based on available memory
def get_optimal_batch_size():
    total_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 1024 * 1024 * 1024 * 16  # 16GB default
    # Use 80% of available memory
    usable_memory = total_memory * 0.8
    # Estimate memory per sample (adjust based on your model)
    memory_per_sample = 1024 * 1024 * 50  # 50MB per sample estimate
    return max(4, int(usable_memory / memory_per_sample))

# Initialize wandb with expanded config
wandb.init(
    project="q-transformer",
    config={
        "num_episodes": 2000,
        "max_steps_per_episode": 200,
        "learning_rate": 3e-4,
        "batch_size": 128, # get_optimal_batch_size(),
        "grad_accum_steps": 8,
        "model_type": "QRoboticTransformer",
        "environment": "MockEnvironment",
        "optimizer": "AdamAtan2",
        "scheduler": "cosine_with_warmup",
        "weight_decay": 1e-2,
        "warmup_steps": 1000,
    }
)

# Setup environment
from q_transformer.mocks import MockEnvironment

env = MockEnvironment(
    state_shape = (3, 6, 224, 224),
    text_embed_shape = (768,)
)

# Setup agent with wandb
agent = Agent(
    get_model(new_model=True),
    environment = env,
    num_episodes = wandb.config.num_episodes,
    max_num_steps_per_episode = wandb.config.max_steps_per_episode,
    epsilon_start = 0.5,
    epsilon_end = 0.01,
    num_steps_to_target_epsilon = 5000,
    use_wandb = True,
    wandb_project = "q-transformer"
)

# Run agent
agent()

# Setup Q-learner with optimized parameters
q_learner = QLearner(
    get_model(new_model=True),
    dataset = ReplayMemoryDataset(),
    num_train_steps = 20000,
    learning_rate = wandb.config.learning_rate,
    batch_size = wandb.config.batch_size,
    grad_accum_every = wandb.config.grad_accum_steps,
    weight_decay = wandb.config.weight_decay,
    max_grad_norm = 1.0,
    checkpoint_dir = './my_training_run',
    save_freq = 200,
    use_wandb = True,
    wandb_project = "q-transformer",
    wandb_run_name = f"training_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    dataloader_kwargs = dict(
        shuffle = True,
        num_workers = 4,
        pin_memory = True,
        prefetch_factor = 2,
    ),
    q_target_ema_kwargs = dict(
        beta = 0.99,
        update_after_step = 100,
        update_every = 10
    ),
)

# Train with monitoring
q_learner()

# Close wandb
wandb.finish()

# For inference, load the best checkpoint
def load_for_inference(checkpoint_dir='./my_training_run'):
    model = get_model(checkpoint_dir)
    model.eval()
    return model

# Load and run inference
inference_model = load_for_inference()
with torch.no_grad():
    video = torch.randn(2, 3, 6, 224, 224)
    instructions = [
        'bring me that apple sitting on the table',
        'please pass the butter'
    ]
    actions = inference_model.get_optimal_actions(video, instructions) 