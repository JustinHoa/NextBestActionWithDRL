import torch

STATE_SIZE = 44  # Gender, marital status, 21 blanks, 21 queue features
ACTION_SIZE = 21

BUFFER_SIZE = 10_000
BATCH_SIZE = 128
GAMMA = 0.99
LR = 1e-4
UPDATE_EVERY = 4
TAU = 1e-3

N_STEP = 3

MAX_T = 100
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.9999
CHECKPOINT_INTERVAL = 10_000

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ACTIVITY_INFO_PATH = "data/raw/activity_info.json"

# Prioritized Experience Replay params
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_BETA_FRAMES = 100_000
PER_EPS = 1e-5

# Rainbow / categorical parameters
RAINBOW_ATOMS = 51
RAINBOW_VMIN = -10.0
RAINBOW_VMAX = 10.0
NOISY_STD_INIT = 0.5
