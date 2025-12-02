from agents.dqn_agent import DQNAgent
from common.utils import ACTION_SIZE, STATE_SIZE


class MultiStepDqnAgent(DQNAgent):
    """
    Cài đặt Multi-step DQN.
    - Kế thừa toàn bộ logic học của DQNAgent.
    - Kích hoạt Multi-step learning bằng cách truyền `n_step` vào BaseAgent.
    """

    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE, seed=0, **kwargs):
        # Gọi init của lớp cha (DQNAgent -> BaseAgent) và bật multi-step
        # n_step=3 là một giá trị phổ biến
        super().__init__(state_size, action_size, seed, n_step=3, **kwargs)