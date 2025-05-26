from collections import namedtuple, deque
import random

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory(object):
    """A simple FIFO experience replay buffer."""
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Saves a transition."""
        # for i, arg in enumerate(args):
        #     if hasattr(arg, 'shape'):
        #         print(f"Transition element {i}: shape={arg.shape}, dtype={getattr(arg, 'dtype', type(arg))}")
        #     else:
        #         print(f"Transition element {i}: type={type(arg)}, value={arg}")
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Samples a batch of transitions."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)