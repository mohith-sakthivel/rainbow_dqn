import numpy as np

# pytorch modules
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as Scheduler

from models import NoisyDistNet
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

    """
    An agent that learns through Rainbow DQN RL Algorithm

    For details refer:
        Rainbow: Combining Improvements in Deep Reinforcement Learning
        [https://arxiv.org/abs/1710.02298]

    Agent has functionality to disable any of the 6 major
    improvements incorporated in rainbow

    Arguments:

    """

    def __init__(self, obs_dim, num_actions, seed=None, n_step=1,
                 n_net=lambda obs, act, atoms: NoisyDistNet(obs, [64, 64, 64], act, atoms),
                 policy_update_freq=2, target_update_freq=75,
                 mini_batch=32, discount=0.999, replay_mem=10000,
                 lr={'start': 0.0005, 'end': 0.0001, 'period': 500},
                 eps={'start': 0.9, 'end': 0.05, 'period': 2500},
                 pri_buf_args={'alpha': 0.7, 'beta': (0.5, 1), 'period': 1e6},
                 distrib_args={'atoms': 21},
                 clip_grads=10, check_pts=[], save_path=None,
                 no_duel=False, no_double=False, no_priority_buf=False,
                 no_noise=False, no_distrib=False):
        self.num_actions = num_actions
        self.obs_dim = obs_dim
        self.n_step = n_step
        self.policy_update_freq = policy_update_freq
        self.target_update_freq = target_update_freq * policy_update_freq
        self.mini_batch = mini_batch
        self.discount = discount
        self.replay_mem = replay_mem
        self.clip_grads = clip_grads
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
        self.no_distrib = no_distrib
        # use gpu if available
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        # set seed for random elementsself.num_actions
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed(seed)
        self.policy_rand_gen = np.random.default_rng(seed)
        # assign distributional value output parameters
        if self.no_distrib is False:
            assert 'min_val' in distrib_args,\
                'Minimum state value is not passed in the arguments'
            assert 'max_val' in distrib_args,\
                'Maximum state value is not passed in the arguments'
            self.min_val = distrib_args['min_val']
            self.max_val = distrib_args['max_val']
            self.num_atoms = distrib_args.get('atoms')
            def network(obs, act): return n_net(obs, act, self.num_atoms)
            self.delta_z =\
                (self.max_val - self.min_val)/(self.num_atoms - 1)
            self.z_atoms = [self.min_val]
            for _ in range(1, self.num_atoms):
                self.z_atoms.append(self.z_atoms[-1] + self.delta_z)
            self.z_atoms = torch.tensor(self.z_atoms, device=self.device,
                                        dtype=torch.float32)
        else:
            network = n_net
        # setup the neural network
        self.policy_net = network(obs_dim, self.num_actions)
        self.target_net = network(obs_dim, self.num_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        # setup experience replay  buffer
        if self.no_priority_buf:
            self.replay = ReplayBuffer(self.replay_mem, self.mini_batch)
        else:
            if pri_buf_args.get('period', None) is not None:
                pri_buf_args['period'] = pri_buf_args['period']/self.policy_update_freq
            self.replay = \
                PrioritizedReplayBuffer(self.replay_mem, self.mini_batch,
                                        **pri_buf_args)
        # setup optimizer
        self.lr = lr
        if isinstance(self.lr, dict):
            self.lr_schedule = True
            self.optimizer = optim.Adam(self.policy_net.parameters(),
                                        lr=self.lr.get('start'))
            if self.lr.get('decay', None) is None:
                decay = (self.lr['end']/self.lr['start']) ** (1/self.lr['period'])
            else:
                decay = self.lr['decay']
            self.lr_scheduler = Scheduler.ExponentialLR(self.optimizer, decay)
        else:
            self.lr_schedule = False
            self.optimizer = optim.Adam(self.policy_net.parameters(),
                                        lr=self.lr, eps=1.5e-4)
        # select action selection function
        self.get_action = self._action_selection_wrappper()
        # select the loss function
        if self.no_distrib:
            self._compute_loss = self._compute_loss_mse
        else:
            self._compute_loss = self._compute_loss_cross_entropy
        # buffer to store the current episode data
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.steps = 0
        self.episodes = 0
        self.episode_reward = 0
        self.n_discounts = [self.discount**i for i in range(self.n_step)]
        self.n_disc = self.discount ** self.n_step
        # training history
        self.episode_rewards_hist = []
        self.loss_hist = []

    def _action_selection_wrappper(self):
        """
        Sets the action selection method based on the agent config
        """
        if self.no_distrib:
            get_max = self._get_max_action
        else:
            get_max = self._get_max_action_distrib

        def eps_greedy_wrapper(obs):
            """ wrapper for greedy action selection """
            if self.policy_rand_gen.random() < self.eps.value:
                # Take random exploratory action
                action = self.policy_rand_gen.integers(0, self.num_actions)
                return action
            return get_max(obs)

        def noisy_wrapper(obs):
            """ wrapper for selecting action from noisy networks """
            self.policy_net.feed_noise()
            return get_max(obs)

        if self.no_noise:
            return eps_greedy_wrapper
        else:
            return noisy_wrapper

    def _choose_max(self, values):
        max_val = values[0]
        max_idxes = [0]
        for i in range(1, values.size):
            if values[i] >= max_val:
                if values[i] == max_val:
                    max_idxes.append(i)
                else:
                    max_val = values[i]
                    max_idxes = [i]
        if len(max_idxes) == 1:
            return np.array(max_idxes, dtype='int64').item()
        else:
            return np.random.choice(max_idxes, 1).item()

    def _get_max_action(self, obs):
        """
        Select an action with epsilon greedy policy

        Arguments:
            obs - an array/list of the agent's observation
        """
        with torch.no_grad():
            obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
            # Take optimal action
            action_vals = self.policy_net(obs).cpu().numpy()
            return self._choose_max(action_vals)

    def _get_max_action_distrib(self, obs):
        """
        Select the max action when the network models
        a distribution over the action-value function

        Arguments:
            obs - an array/list of the agent's observation
        """
        with torch.no_grad():
            obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
            # Take optimal action
            action_probs = self.policy_net(obs)
            action_vals = torch.sum((action_probs * self.z_atoms), dim=-1)
            return self._choose_max(action_vals.cpu().numpy())

    def start(self, obs):
        """
        Start the agent for the episode

        Arguments:
            obs - an array/list of the agent's observation
        """
        self.episodes += 1
        self.episode_reward = 0
        self.obs_buffer.append(obs)
        action = self.get_action(obs)
        self.action_buffer.append(action)
        return action

    def take_step(self, reward, obs):
        """
        Get an action from the agent for the current observation

        Arguments:
            reward - reward recieved from the environment
            obs    - an array/list of the agent's observation
        """
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
            self._compute_loss()
            self._optimize()
            # update target network
            if self.steps % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        return action

    def end(self, reward):
        """Reset the agent for the next episode

        Arguments:
            reward - reward recieved from the environment
        """
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
            self._compute_loss()
            self._optimize()
            # update target network
            if self.steps % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        return self.episode_reward

    def _optimize(self):
        """ Optimize the model parameters for current gradient """
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

    def _compute_loss_cross_entropy(self):
        """ Compute the loss and backpropogate gradients """
        # Sample data from replay buffer
        data, is_weights, idxes = self.replay.sample()
        obs, actions, next_obs, rewards, t_masks = data
        # convert data into tensors
        obs = torch.tensor(obs, device=self.device,
                           dtype=torch.float32)
        rewards = torch.tensor(rewards, device=self.device,
                               dtype=torch.float32).unsqueeze(-1)
        next_obs = torch.tensor(next_obs, device=self.device,
                                dtype=torch.float32)[t_masks]
        t_masks_rev = [not val for val in t_masks]
        # feed noise into the networks
        if self.no_noise is False:
            self.policy_net.feed_noise()
            self.target_net.feed_noise()
        # compute target probs
        with torch.no_grad():
            # find max_actions
            if self.no_double:
                next_obs_probs = self.target_net(next_obs)
                if self.no_noise is False:
                    self.target_net.feed_noise()
            else:
                next_obs_probs = self.policy_net(next_obs)
                if self.no_noise is False:
                    self.policy_net.feed_noise()
            max_actions = \
                (next_obs_probs * self.z_atoms).sum(dim=-1).max(dim=-1)[1]
            # calculate target value atoms
            target = rewards[t_masks] + self.n_disc * self.z_atoms
            target.clamp_(self.min_val, self.max_val)
            # project non-terminal state targets on to the support vectors
            b = (target - self.min_val)/self.delta_z
            low = torch.floor(b).type(torch.int64)
            up = torch.ceil(b).type(torch.int64)
            # project terminal state targets on to the support vectors
            b_term = (rewards[t_masks_rev].squeeze(-1) - self.min_val)/self.delta_z
            low_term = torch.floor(b_term).type(torch.int64)
            up_term = torch.ceil(b_term).type(torch.int64)
            # calulate next state-action value probability
            nxt_obs_no = [i for i in range(next_obs.shape[0])]
            next_probs = self.target_net(next_obs)[nxt_obs_no, max_actions, :]
            # Distribute the probability of non-terminal targets
            target_probs = torch.zeros((self.mini_batch, self.num_atoms),
                                       device=self.device, dtype=torch.float32)
            for idx in range(self.num_atoms):
                target_probs[t_masks, low[:, idx]] += \
                    next_probs[nxt_obs_no, idx] * (up[:, idx] - b[:, idx])
                target_probs[t_masks, up[:, idx]] += \
                    next_probs[nxt_obs_no, idx] * (b[:, idx] - low[:, idx])
                target_probs[t_masks, low[:, idx]] += \
                    next_probs[nxt_obs_no, idx] * (up[:, idx] == low[:, idx]).int()
            # Distribute the probability of terminal targets
            target_probs[t_masks_rev, low_term] += (up_term - b_term)
            target_probs[t_masks_rev, up_term] += (b_term - low_term)
            target_probs[t_masks_rev, low_term] += (up_term == low_term).int()
        # calculate value distribution of the state-action pair
        sample_no = [i for i in range(self.mini_batch)]
        obs_act_prob = self.policy_net(obs)[sample_no, actions, :]
        # compute cross entropy loss
        td_error = torch.sum(-target_probs * torch.log(obs_act_prob), dim=-1)
        loss = td_error.sum(dim=-1)
        # compute gradients
        self.optimizer.zero_grad()
        loss.backward()
        if self.steps % 1000 == 0:
            self.loss_hist.append(loss.item())
        # perform importance sampling and update td_error in buffer
        if self.no_priority_buf is False:
            if is_weights is not None:
                is_weights = torch.tensor(is_weights, device=self.device,
                                          dtype=torch.float32)
                self.replay.update(idxes, td_error.cpu().detach().numpy())
                td_error = td_error * is_weights
            else:
                self.replay.update(idxes, td_error)

    def _compute_loss_mse(self):
        """ Compute the loss and backpropogate gradients """
        # Sample data from replay buffer
        data, is_weights, idxes = self.replay.sample()
        obs, actions, next_obs, rewards, t_masks = data
        # convert data into tensors
        obs = torch.tensor(obs, device=self.device,
                           dtype=torch.float32)
        actions = torch.tensor(actions, device=self.device,
                               dtype=torch.int64).unsqueeze(-1)
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
            targets = rewards + self.n_disc * nxt_obs_vals
            td_error = (targets-obs_vals)
        # compute loss
        loss = -torch.sum(td_error * obs_vals)
        # compute gradients
        self.optimizer.zero_grad()
        loss.backward()
        if self.steps % 1000 == 0:
            self.loss_hist.append(loss.item())
        # perform importance sampling and update td_error in buffer
        if self.no_priority_buf is False:
            if is_weights is not None:
                is_weights = torch.tensor(is_weights, device=self.device,
                                          dtype=torch.float32)
                self.replay.update(idxes, td_error.cpu().numpy())
                td_error = td_error * is_weights
            else:
                self.replay.update(idxes, td_error)

    def get_train_data(self):
        """ Return last training data """
        if len(self.loss_hist) != 0:
            return {'loss': self.loss_hist[-1]}
        else:
            return None
