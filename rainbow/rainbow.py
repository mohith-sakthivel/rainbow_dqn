import random
import numpy as np
from collections import deque, namedtuple

# pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as Scheduler


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state',
                         'reward', 'td_error'))


class ReplayBuffer():
    """
    Stores the transitions experienced by the agent
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.experience_buffer = deque(maxlen=self.capacity)

    def __len__(self):
        return len(self.experience_buffer)

    def add_item(self, item):
        self.experience_buffer.append(item)

    def sample_batch(self, batch_size):
        return random.sample(self.experience_buffer, batch_size)


class EpsilonScheduler():
    """
    Scheduler for epsilon during the learning process
    """
    def __init__(self, initial, final, period):
        self.initial = initial
        self.final = final
        self.period = period
        self.decay_factor = np.power((self.final/self.initial), 1/self.period)
        self.curr_epsilon = self.initial

    def get_epsilon(self):
        """ Get the current value of epsilon"""
        epsilon = self.curr_epsilon
        self.step()
        return epsilon

    def step(self):
        """ Decrease epsilon by a step """
        if self.curr_epsilon > self.final:
            self.curr_epsilon = self.curr_epsilon * self.decay_factor


class DuelNet(nn.Module):
    def __init__(self, num_states, hid_lyrs, num_actions, activation=nn.ReLU):
        super().__init__()
        layers = [num_states] + hid_lyrs
        nn_layers = []
        for index in range(len(layers)-1):
            nn_layers.append(nn.Linear(layers[index], layers[index+1]))
            nn_layers.append(activation())
        self.shared_net = nn.Sequential(*nn_layers)
        self.value = nn.Linear(hid_lyrs[-1], 1)
        self.advantage = nn.Linear(hid_lyrs[-1], num_actions)

    def forward(self, x):
        shared_net = self.shared_net(x)
        value = self.value(shared_net)
        advantage = self.advantage(shared_net)
        action_values = value + (advantage - advantage.mean(-1, keepdim=True))
        return action_values


class NormalNet(nn.Module):
    def __init__(self, num_states, hid_lyrs, num_actions, activation=nn.ReLU):
        super().__init__()
        layers = [num_states] + hid_lyrs
        nn_layers = []
        for index in range(len(layers)-1):
            nn_layers.append(nn.Linear(layers[index], layers[index+1]))
            nn_layers.append(activation())
        nn_layers.append(nn.Linear(layers[-1], num_actions))
        self.shared_net = nn.Sequential(*nn_layers)

    def forward(self, x):
        return self.shared_net(x)


class Rainbow():

    def __init__(self, state_space, num_actions, activation=nn.ReLU,
                 hid_lyrs=[64, 64], target_update_freq=150,
                 mini_batch=32, discount=0.999, replay_mem=10000,
                 max_lr=None, lr=0.001, anneal_period=2500,
                 eps_ini=0.9, eps_fin=0.05, explo_period=1000,
                 seed=None,
                 no_duel=False, no_double=False):
        self.num_actions = num_actions
        self.state_space = state_space
        self.activation = activation
        self.target_update_freq = target_update_freq
        self.policy_update_freq = 2
        self.lr = lr
        self.max_lr = max_lr
        self.discount = discount
        self.mini_batch = mini_batch
        self.eps_schdlr = EpsilonScheduler(eps_ini, eps_fin, explo_period)
        # flags for disabling rainbow components
        self.no_duel = no_duel
        self.no_double = no_double
        # Check for gpu
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        # set seed for agent
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed(seed)
        self.policy_rand_generator = np.random.default_rng(seed)
        # setup the neural network and replay buffer
        if self.no_duel:
            neural_net = NormalNet
        else:
            neural_net = DuelNet
        self.policy_net = neural_net(state_space, hid_lyrs,
                                     num_actions, self.activation)
        self.target_net = neural_net(state_space, hid_lyrs,
                                     num_actions, self.activation)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.replay = ReplayBuffer(10000)
        if self.max_lr is not None:
            assert max_lr > lr, "max lr is less than final lr"
            self.optimizer = optim.Adam(self.policy_net.parameters(),
                                        lr=self.max_lr)
            decay = np.power((self.lr/self.max_lr), 1/anneal_period)
            self.lr_scheduler = Scheduler.ExponentialLR(self.optimizer, decay)
        else:
            self.optimizer = optim.Adam(self.policy_net.parameters(),
                                        lr=self.lr)
        # buffer to store the current episode data
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.steps = 0
        self.episodes = 0
        self.episode_reward = 0
        # training history
        self.episode_rewards_hist = []
        self.loss_hist = []

    def get_action(self, state):
        """ Select an action as per agent policy """
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        if self.policy_rand_generator.random() < self.eps_schdlr.get_epsilon():
            # Take random exploratory action
            action = [self.policy_rand_generator.integers(0, self.num_actions)]
            return torch.tensor(action, device=self.device,
                                dtype=torch.int32)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(-1)[1].view(1, 1)

    def start(self, state):
        """ Start the agent for the episode """
        self.episodes += 1
        self.episode_reward = 0
        self.state_buffer.append(state)
        action = self.get_action(state)
        self.action_buffer.append(action)
        return action.item()

    def take_step(self, reward, state):
        """ Get an action from the agent for the current observation """
        self.steps += 1
        self.episode_reward += reward
        self.reward_buffer.append(reward)
        self.replay.add_item(Transition(self.state_buffer[-1],
                                        self.action_buffer[-1],
                                        state, reward, None))
        self.state_buffer.append(state)
        action = self.get_action(state)
        self.action_buffer.append(action)
        # update policy network
        if (self.steps % self.policy_update_freq == 0 and
                self.steps >= self.mini_batch):
            self.compute_grad()
        # update target network
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return action.item()

    def end(self, reward):
        """ Reset the agent for the next episode """
        self.steps += 1
        self.episode_reward += reward
        self.episode_rewards_hist.append(self.episode_reward)
        self.reward_buffer.append(reward)
        self.replay.add_item(Transition(self.state_buffer[-1],
                                        self.action_buffer[-1],
                                        None, reward, None))
        # clear stored memory
        self.reward_buffer.clear()
        self.state_buffer.clear()
        self.action_buffer.clear()
        # update policy network
        if (self.steps % self.policy_update_freq == 0 and
                self.steps >= self.mini_batch):
            self.compute_grad()
        # update target network
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return self.episode_reward

    def compute_grad(self):
        """ Compute the gradients """
        sample = self.replay.sample_batch(self.mini_batch)
        batch = Transition(*zip(*sample))
        # unpack data into tensors
        states = torch.tensor(batch.state, device=self.device,
                              dtype=torch.float32)
        actions = torch.tensor(batch.action, device=self.device,
                               dtype=torch.int64)
        actions.unsqueeze_(-1)
        rewards = torch.tensor(batch.reward, device=self.device,
                               dtype=torch.float32)
        terminal_mask = torch.tensor(tuple(map(lambda x: x is not None,
                                               batch.next_state)),
                                     device=self.device, dtype=torch.bool)
        next_states = \
            torch.tensor([s for s in batch.next_state if s is not None],
                         device=self.device, dtype=torch.float32)
        next_states.squeeze_()
        # compute current state-action value
        st_act_vals = self.policy_net(states).gather(-1, actions).squeeze(-1)
        # compute target
        nxt_st_act_vals = torch.zeros_like(terminal_mask, device=self.device,
                                           dtype=torch.float32)
        with torch.no_grad():
            # calculate argmax of the next state's action-value
            if self.no_double:
                max_actions = \
                    self.target_net(next_states).max(-1, keepdim=True)[1]
            else:
                max_actions = \
                    self.policy_net(next_states).max(-1, keepdim=True)[1]
            # calculate the action-value from the target network
            nxt_st_act_vals[terminal_mask] += \
                torch.gather(self.target_net(next_states),
                             -1, max_actions).squeeze(-1)
        targets = rewards + self.discount * nxt_st_act_vals
        # compute Huber loss
        loss = F.mse_loss(st_act_vals, targets, reduction='mean')
        # compute gradients
        self.optimizer.zero_grad()
        loss.backward()
        # optimize the policy network
        self.optimizer.step()
        if self.max_lr and self.lr_scheduler.get_last_lr()[0] > self.lr:
            self.lr_scheduler.step()
        self.loss_hist.append(loss)

    def get_train_data(self):
        return {'loss': self.loss_hist[-1]}
