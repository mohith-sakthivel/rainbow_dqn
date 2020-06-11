"""
This module contains an implementation of a buffer for
Prioritized Experience Replay with DQN agents

Paper: https://arxiv.org/abs/1511.05952
"""


class PrioritizedReplayBuffer():
    """
    Prioritized Buffer for storing transitions

    Details:
    Ranks based implementation is used.
    The cumulative probability distribution function is divided
    into k segments of equal probability with each segment having
    different number of transitions.
    """

    def __init__(self, capacity, alpha, beta, segments):
        self.capacity = capacity
        self._buffer = []
        self.alpha = alpha
        self.beta = beta
        self.segments = segments
