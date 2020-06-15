import numpy as np

# pytorch modules
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as Scheduler

from models import DuelNet
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, Transition
from schedules import ExponentialScheduler


class ValueVar():
    __slots__ = ['value']

    def __init__(self, value):
        self.value = value


class ExpScheduleAutoStep(ExponentialScheduler):
    """
    Scheduler for epsilon during the learning process
    Steps schedule automatically during each value request
    """
    @property
    def value(self):
        """ Returns currendt variable var """
        val = self._value
        self.step()
        return val


class Rainbow():

    def __init__(self, obs_dim, num_actions, seed=None, n_step=1,
                 network=lambda obs, act: DuelNet(obs, [64, 64, 64], act),
                 policy_update_freq=2, target_update_freq=150,
                 mini_batch=32, discount=0.999, replay_mem=10000,
                 lr={'start': 0.0005, 'end': 0.0001, 'period': 500},
                 eps={'start': 0.9, 'end': 0.05, 'period': 2500},
                 pri_buf_args={'alpha': 0.7, 'beta': (0.5, 1), 'period': 1e6},
                 clip_grads=10, check_pts=[], save_path=None,
                 no_duel=False, no_double=False, no_priority_buf=False,
                 no_noise=False):
        self.num_actions = num_actions
        self.obs_dim = obs_dim
        self.n_step = n_step
        self.target_update_freq = target_update_freq
        self.policy_update_freq = policy_update_freq
        self.discount = discount
        self.mini_batch = mini_batch
        self.replay_mem = replay_mem
        # set epsilon
        if isinstance(eps, dict):
            self.eps = ExpScheduleAutoStep(**eps)
        else:
            self.eps = ValueVar(eps)
        # flags for disabling rainbow components
        self.no_duel = no_duel
        self.no_double = no_double
        self.no_priority_buf = no_priority_buf
        self.no_noise = no_noise
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
        # setup the neural network
        self.policy_net = network(obs_dim, num_actions)
        self.target_net = network(obs_dim, num_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.clip_grads = clip_grads
        # setup experience replay  buffer
        if self.no_priority_buf:
            self.replay = ReplayBuffer(self.replay_mem, self.mini_batch)
        else:
            if pri_buf_args.get('period', None) is not None:
                pri_buf_args['period'] = \
                    pri_buf_args['period']/self.policy_update_freq
            self.replay = \
                PrioritizedReplayBuffer(self.replay_mem, self.mini_batch,
                                        **pri_buf_args)
        # setup optimizer
        self.lr = lr
        if isinstance(self.lr, dict):
            self.lr_schedule = True
            self.optimizer = optim.Adam(self.policy_net.parameters(),
                                        lr=self.lr.get('start'))
            decay = ((self.lr.get('end') / self.lr.get('start')) **
                     (1/self.lr.get('period')))
            self.lr_scheduler = Scheduler.ExponentialLR(self.optimizer, decay)
        else:
            self.lr_schedule = False
            self.optimizer = optim.Adam(self.policy_net.parameters(),
                                        lr=self.lr)
        # buffer to store the current episode data
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.steps = 0
        self.episodes = 0
        self.episode_reward = 0
        self.n_discounts = [self.discount**i for i in range(self.n_step)]
        # training history
        self.episode_rewards_hist = []
        self.loss_hist = []

    def get_action(self, obs):
        """ Select an action as per agent policy """
        with torch.no_grad():
            obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
            # feed noise into the networks
            if self.no_noise is False:
                self.policy_net.feed_noise()
                # Take optimal action
                return self.policy_net(obs).max(-1, keepdims=True)[1].cpu().numpy()
            if self.policy_rand_generator.random() < self.eps.value:
                # Take random exploratory action
                action = [self.policy_rand_generator.integers(0, self.num_actions)]
                return np.array(action, dtype=np.int32)
            else:
                # Take optimal action
                return self.policy_net(obs).max(-1, keepdims=True)[1].cpu().numpy()

    def start(self, obs):
        """ Start the agent for the episode """
        self.episodes += 1
        self.episode_reward = 0
        self.obs_buffer.append(obs)
        action = self.get_action(obs)
        self.action_buffer.append(action)
        return action.item()

    def take_step(self, reward, obs):
        """ Get an action from the agent for the current observation """
        self.steps += 1
        self.episode_reward += reward
        self.reward_buffer.append(reward)
        n_reward = 0
        # Add transition to replay buffer
        if len(self.reward_buffer) >= self.n_step:
            for rew, disc in zip(self.reward_buffer[-self.n_step:],
                                 self.n_discounts):
                n_reward += rew*disc
            self.replay.add(Transition(self.obs_buffer[-self.n_step],
                                       self.action_buffer[-self.n_step],
                                       obs, n_reward, True, np.inf))
        self.obs_buffer.append(obs)
        action = self.get_action(obs)
        self.action_buffer.append(action)
        # update policy network
        if (self.steps % self.policy_update_freq == 0 and
                len(self.replay) >= self.mini_batch):
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
        n_trans = min(len(self.reward_buffer), self.n_step)
        for i in range(1, n_trans+1):
            n_reward = 0
            for rew, disc in zip(self.reward_buffer[-i:],
                                 self.n_discounts[:i]):
                n_reward += rew*disc
            self.replay.add(Transition(self.obs_buffer[-i],
                                       self.action_buffer[-i],
                                       np.zeros(self.obs_dim, dtype='float32'),
                                       n_reward, False, np.inf))
        # clear stored memory
        self.reward_buffer.clear()
        self.obs_buffer.clear()
        self.action_buffer.clear()
        # update policy network
        if (self.steps % self.policy_update_freq == 0 and
                len(self.replay) >= self.mini_batch):
            self.compute_grad()
        # update target network
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return self.episode_reward

    def compute_grad(self):
        """ Compute the gradients """
        data, is_weights, idxes = self.replay.sample()
        obs, actions, next_obs, rewards, t_masks = data
        # convert data into tensors
        obs = torch.tensor(obs, device=self.device,
                           dtype=torch.float32)
        actions = torch.tensor(actions, device=self.device,
                               dtype=torch.int64)
        rewards = torch.tensor(rewards, device=self.device,
                               dtype=torch.float32)
        next_obs = torch.tensor(next_obs, device=self.device,
                                dtype=torch.float32)[t_masks]
        # feed noise into the networks
        if self.no_noise is False:
            self.policy_net.feed_noise()
            self.target_net.feed_noise()
        # compute current state-action value
        obs_vals = self.policy_net(obs).gather(-1, actions).squeeze(-1)
        # compute target
        nxt_obs_vals = torch.zeros_like(rewards, device=self.device,
                                        dtype=torch.float32)
        with torch.no_grad():
            # calculate argmax of the next state's action-value
            if self.no_double:
                max_act = self.target_net(next_obs).max(-1, keepdim=True)[1]
                if self.no_noise is False:
                    self.target_net.feed_noise()
            else:
                if self.no_noise is False:
                    self.policy_net.feed_noise()
                max_act = self.policy_net(next_obs).max(-1, keepdim=True)[1]
            # calculate the action-value from the target network
            nxt_obs_vals[t_masks] += \
                torch.gather(self.target_net(next_obs), -1, max_act).squeeze(-1)
            targets = rewards + self.discount * nxt_obs_vals
            td_error = (targets-obs_vals)
        # perform importance sampling and update td_error in buffer
        if self.no_priority_buf is False:
            if is_weights is not None:
                is_weights = torch.tensor(is_weights, device=self.device,
                                          dtype=torch.float32)
                self.replay.update(idxes, td_error)
                td_error = td_error * is_weights
            else:
                self.replay.update(idxes, td_error)
        # compute loss
        loss = -torch.sum(td_error * obs_vals)
        # compute gradients
        self.optimizer.zero_grad()
        loss.backward()
        if self.no_duel is False:
            # rescale of shared layers
            for param in self.policy_net.shared_net.parameters():
                if param.requires_grad:
                    param.grad.data.div_(np.sqrt(2))
            # clip gradient norms
            if self.clip_grads is not None:
                for param in self.policy_net.parameters():
                    if param.requires_grad:
                        scale = min(self.clip_grads/param.grad.data.norm(),
                                    self.clip_grads)
                        param.grad.data.mul_(scale)
        # optimize the policy network
        self.optimizer.step()
        if (self.lr_schedule and
                self.lr_scheduler.get_last_lr()[0] > self.lr.get('end')):
            self.lr_scheduler.step()
        if self.steps % 1000 == 0:
            self.loss_hist.append(loss.item())

    def get_train_data(self):
        """ Return last training data"""
        if len(self.loss_hist) != 0:
            return {'loss': self.loss_hist[-1]}
        else:
            return None
