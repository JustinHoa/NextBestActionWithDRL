from src.agents.dqn_agent import DQNAgent
from src.config import (ACTION_SIZE, BATCH_SIZE, BUFFER_SIZE, GAMMA, N_STEP,
                        STATE_SIZE)
from src.replay.multi_step_replay_buffer import MultiStepReplayBuffer


class MultiStepDQNAgent(DQNAgent):
    """DQN agent trained with n-step returns."""

    def __init__(self,
                 state_size: int = STATE_SIZE,
                 action_size: int = ACTION_SIZE,
                 seed: int = 0,
                 n_step: int = N_STEP):
        super().__init__(state_size=state_size, action_size=action_size, seed=seed)
        self.memory = MultiStepReplayBuffer(buffer_size=BUFFER_SIZE,
                                            batch_size=BATCH_SIZE,
                                            n_step=n_step,
                                            gamma=GAMMA,
                                            seed=seed)
