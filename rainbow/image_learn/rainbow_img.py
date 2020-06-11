import random
import numpy as np
from collections import deque, namedtuple

# pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
    def __init__(self, num_states, hid_lyrs, num_actions):
        super().__init__()
        layers = [num_states] + hid_lyrs
        nn_layers = []
        for index in range(len(layers)-1):
            nn_layers.append(nn.Linear(layers[index], layers[index+1]))
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
    def __init__(self, num_states, hid_lyrs, num_actions):
        super().__init__()
        layers = [num_states] + hid_lyrs + [num_actions]
        nn_layers = []
        for index in range(len(layers)-1):
            nn_layers.append(nn.Linear(layers[index], layers[index+1]))
        self.shared_net = nn.Sequential(*nn_layers)

    def forward(self, x):
        return self.shared_net(x)


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size-(kernel_size-1)-1)//stride+1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)
        # self.head = nn.Linear(linear_input_size, 16)
        # self.heada = nn.Linear(16, outputs)
        # self.headb = nn.Linear(16, 1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
        # x = self.head(x.view(x.size(0), -1))
        # advantage = self.heada(x)
        # value = self.headb(x)
        # return value + (advantage - advantage.mean(-1, keepdim=True))


class Rainbow():

    def __init__(self, state_space, num_actions, hid_lyrs=[16, 16],
                 target_update_freq=10, mini_batch=128, lr=0.02,
                 replay_mem=10000, seed=None, eps_ini=0.9, eps_fin=0.1,
                 explo_period=200, discount=0.999):
        self.num_actions = num_actions
        self.state_space = state_space
        self.target_update_freq = target_update_freq
        self.lr = lr
        self.discount = discount
        self.mini_batch = mini_batch
        self.eps_schdlr = EpsilonScheduler(eps_ini, eps_fin, explo_period)
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
        self.policy_net = DQN(state_space[0], state_space[1], num_actions)
        self.target_net = DQN(state_space[0], state_space[1], num_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(),
                                       lr=self.lr)
        self.replay = ReplayBuffer(10000)
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
        if self.steps % self.mini_batch == 0:
            self.compute_grad()
            # update target network
            if self.steps % (self.target_update_freq * self.mini_batch) == 0:
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
        if self.steps % self.mini_batch == 0:
            self.compute_grad()
            # update target network
            if self.steps % (self.target_update_freq * self.mini_batch) == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        return self.episode_reward

    def compute_grad(self):
        """ Compute the gradients """
        sample = self.replay.sample_batch(self.mini_batch)
        batch = Transition(*zip(*sample))
        # unpack data into tensors
        states = torch.tensor(batch.state, device=self.device,
                              dtype=torch.float32)
        states.squeeze_()
        actions = torch.tensor(batch.action, device=self.device,
                               dtype=torch.int64)
        actions.unsqueeze_(-1)
        rewards = torch.tensor(batch.reward, device=self.device,
                               dtype=torch.float32)
        terminal_mask = torch.tensor(tuple(map(lambda x: x is not None,
                                               batch.next_state)),
                                     device=self.device, dtype=torch.bool)
        next_states = torch.tensor([s for s in batch.next_state if s is not None],
                                   device=self.device, dtype=torch.float32)
        next_states.squeeze_()
        # compute current state-action value
        st_act_vals = self.policy_net(states).gather(-1, actions).squeeze(-1)
        # compute target
        nxt_st_act_vals = torch.zeros_like(terminal_mask, device=self.device,
                                           dtype=torch.float32)
        with torch.no_grad():
            # calculate argmax of action-value from the policy network
            max_actions = \
                self.policy_net(next_states).max(-1, keepdim=True)[1]
            # calculate the action-value from the target network
            nxt_st_act_vals[terminal_mask] += \
                self.target_net(next_states).gather(-1, max_actions).squeeze(-1)
        targets = rewards + self.discount * nxt_st_act_vals
        # compute Huber loss
        loss = F.smooth_l1_loss(st_act_vals, targets, reduction='mean')
        # compute gradients
        self.optimizer.zero_grad()
        loss.backward()
        # clip the gradients
        for param_groups in self.optimizer.param_groups:
            for params in param_groups['params']:
                params.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.loss_hist.append(loss)

    def get_train_data(self):
        data = {}
        data["rewards"] = np.mean(self.episode_rewards_hist[-1])
        data["loss"] = self.loss_hist[-1]
        # print("***************************************")
        # print("Episode No : {}".format(self.episodes))
        # print("Avergae Rewards: ", data.get('rewards'))
        # print("Loss: ", data.get("loss"))
        # print("***************************************")
        return data
