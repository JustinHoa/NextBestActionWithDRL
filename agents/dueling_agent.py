import torch
from agents.ddqn_agent import DDQNAgent
from networks.duel_net import DuelingQNetwork
from common.utils import DEVICE, STATE_SIZE, ACTION_SIZE

class DuelingAgent(DDQNAgent):
    """
    Dueling DDQN Agent.
    Kế thừa logic học của DDQN nhưng thay thế kiến trúc mạng.
    """
    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE, seed=0, **kwargs):
        super().__init__(state_size, action_size, seed, **kwargs)
        
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(DEVICE)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=1e-4)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())