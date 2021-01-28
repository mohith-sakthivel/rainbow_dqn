"""
This module contains an implementation of a buffer for
Prioritized Experience Replay with DQN agents

Paper: https://arxiv.org/abs/1511.05952
"""
import heapq
import random

from collections import deque
from rainbow.schedules import LinearScheduler
from copy import deepcopy


class Transition():
    """
    Object for storing transition

    Arguments:
        obs           - First observation of the transition
        action        - Action performed
        next_obs      - Obervations following the action.
                        Number of observatios depends on bootstrapping used.
        rewards       - Rewards recieved following the action.
                        Number of rewards depends on bootstrapping used.
        t_mask        - Mask to identify terminal states
        td_error      - Last calculated td_error of the transition
    """

    __slots__ = ['obs', 'action', 'next_obs', 'rewards', 't_mask', 'td_error']

    def __init__(self, obs, action, next_obs, rewards, terminal_mask,
                 td_error):
        self.obs = obs
        self.action = action
        self.next_obs = next_obs
        self.rewards = rewards
        self.t_mask = terminal_mask
        self.td_error = td_error

    def __lt__(self, value):
        return self.td_error < value.td_error

    def __str__(self):
        string = ''
        string += 'Observation  : ' + str(self.obs) + '\n'
        string += 'Action       : ' + str(self.action) + '\n'
        string += 'Next Obs     : ' + str(self.next_obs) + '\n'
        string += 'Rewards      : ' + str(self.rewards) + '\n'
        string += 'NonT-erminal : ' + str(self.t_mask) + '\n'
        string += 'TD Error     : ' + str(self.td_error) + '\n'
        return string


class ReplayBuffer():
    """
    Buffer to store transitions experienced by the agent
    """

    def __init__(self, capacity, batch_size):
        assert capacity > 0, 'Capacity should be positive'
        assert batch_size <= capacity, 'Batch size' +\
            'should not be greater than the buffer capcity'
        self._capacity = capacity
        self._len = 0
        self._buffer = deque(maxlen=self._capacity)
        self._batch_size = batch_size

    def __len__(self):
        return self._len

    def _unpack_samples(self, idxes):
        """ Unpacks the batch of transition objects """
        obs, actions, next_obs, rewards, t_masks = [], [], [], [], []
        for idx in idxes:
            data = self._buffer[idx]
            obs.append(data.obs)
            actions.append(data.action)
            next_obs.append(data.next_obs)
            rewards.append(data.rewards)
            t_masks.append(data.t_mask)
        return (obs, actions, next_obs, rewards, t_masks)

    def add(self, item):
        if self._len < self._capacity:
            self._len += 1
        self._buffer.append(item)

    def _uniform_sample(self):
        """ Sample uniformly from the buffer """
        sample_idxes = random.sample(range(self._len), self._batch_size)
        return self._unpack_samples(sample_idxes), sample_idxes

    def sample(self):
        data, _ = self._uniform_sample()
        return data, None, None


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Buffer for storing transitions.
    Samples transitions with a stochastic priority

    Details:
        * Ranks based implementation is used.
        * Binary Tree heap is used as an approximation of a sorted array
        * The cumulative probability distribution function is divided
          into k segments of equal probability.
        * A transition is sampled uniformly from each segment.

    Arguments:
        capacity   - capacity of the replay buffer
        alpha      - parameter to emphasise on the priorities.
                     sampling is unifrom if alpha = 0
        beta       - exponent to compensate bias
        batch_size - number of transitions per sample
    """

    def __init__(self, capacity, batch_size,
                 alpha=0.7, beta=(0.5, 1), period=None, freq=20):
        assert capacity > 0, 'Capacity should be positive'
        assert batch_size <= capacity, \
            'Batch size should not be greater than the buffer capcity'
        assert alpha >= 0, 'Alpha should be positive'
        self._capacity = capacity
        self._batch_size = batch_size
        self._buffer = []
        self._alpha = alpha
        if len(beta) == 2:
            if period is None:
                self._period = 10 * self._capacity
            self._sch_freq = 10
            self._sch_steps = 0
            self._schedule_beta = True
            self._period = period
            self._beta = beta[0]
            self._beta_sch = \
                LinearScheduler(beta[0], beta[1],
                                period=self._period/self._sch_freq)
        else:
            self._schedule_beta = False
            self._beta = beta
        self._len = 0
        self._steps = 0
        self._sort_freq = self._capacity
        self._parts = self._get_parts()
        self._is_weights = self._get_is_weights()

    def add(self, item):
        """
        Add an element to the heap

        Arguments:
            item - transition to be added
        """
        if self._len >= self._capacity:
            heapq.heapreplace(self._buffer, item)
        else:
            heapq.heappush(self._buffer, item)
            self._len += 1

    def _get_parts(self):
        """
        Calculates segment boundaries for the given buffer capacity and alpha
        """
        prob_per_seg = 1 / self._batch_size
        partitions = [0, ]
        p_vals = [(1/i)**self._alpha for i in range(self._capacity, 0, -1)]
        p_sum = sum(p_vals)
        probs = [value/p_sum for value in p_vals]
        seg_prob = 0
        for i in range(self._capacity):
            if seg_prob + probs[i] > prob_per_seg:
                # check which side of the partition gives min error
                partitions.append(i)
                seg_prob = probs[i]
                # check if sufficent partitions are made
                if len(partitions) == self._batch_size:
                    break
            else:
                seg_prob += probs[i]
        partitions.append(self._capacity)
        assert len(partitions) == (self._batch_size) + 1  # to be removed
        return partitions

    def _get_is_weights(self):
        is_weights = []
        max_val = 0
        for i in range(self._batch_size):
            value = (((self._parts[i+1]-self._parts[i])/self._capacity) **
                     self._beta)
            is_weights.append(value)
            if value > max_val:
                max_val = value
        is_weights = [val / max_val for val in is_weights]
        return is_weights

    def _update_beta(self):
        """ Update beta with scheduler """
        self._sch_steps += 1
        # update beta at a lowered frequency to reduce compute load
        if self._sch_steps % self._sch_freq == 0:
            self._beta_sch.step()
            self._beta = self._beta_sch.value
            self._is_weights = self._get_is_weights()

    def _priority_sample(self):
        """ Sample from the buffer with priority """
        sample_idxes = []
        for i in range(self._batch_size):
            sample_idxes.append(random.randint(self._parts[i],
                                               self._parts[i+1]-1))
        return self._unpack_samples(sample_idxes), sample_idxes

    def sample(self):
        """
        Sample transitions form the buffer
        """
        if self._steps % self._sort_freq == 0 and self._steps != 0:
            self._buffer.sort()
        self._steps += 1
        if self._len < self._capacity:
            data, idxes = self._uniform_sample()
            return data, None, idxes
        else:
            data, idxes = self._priority_sample()
            is_weights = deepcopy(self._is_weights)
            if self._schedule_beta:
                self._update_beta()
            return data, is_weights, idxes

    def update(self, idxes, td_errors):
        """
        Update the TD errors of the transitions in the buffer

        Arguments:
            idexes   - Indices of the transitions in the current buffer
            td_error - Updated td_errors for those transitions
        """
        assert len(idxes) == len(td_errors), \
            'indexes and corresponding values do not have same length'
        # prioritize over absolute error
        td_errors = abs(td_errors)
        for idx, td_err in zip(idxes, td_errors):
            old_error = self._buffer[idx].td_error
            self._buffer[idx].td_error = td_err
            # update the position of the transition in the queue
            if td_err < old_error:
                heapq._siftdown(self._buffer, 0, idx)
            else:
                heapq._siftup(self._buffer, idx)
