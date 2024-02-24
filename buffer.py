import numpy as np
from collections import deque

class Buffer:
    def __init__(self, params: dict) -> None:
        self.buffer = deque()
        self.size = params["BUFFER_SIZE"]
        self.batch_size = params["BATCH_SIZE"]

    def append(self, obs: np.ndarray, reward: float) -> None:
        self.buffer.append((obs, reward))
        if len(self.buffer) >= self.size:
            self.buffer.popleft()
    
    def get_random_batch(self) -> None:
        indices = np.random.randint(0, len(self.buffer), size=self.batch_size)
        data = self.buffer[indices]
        obv, reward = [d[0] for d in data], [d[1] for d in data]
        return np.array(obv), np.array(reward)
