import numpy as np

# pytorch modules
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as Scheduler

from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, Transition
from schedules import ExponentialScheduler

from datetime import datetime
import os


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
    num_actions        - number of actions available for the agent to take
    n_net              - a function that takes in num_actions (and num_atoms
                         for ditributional networks) as argument to
                         instantiate a pytorch module
    seed               - seed for the agent
    policy_update_freq - number of time steps between each policy
                         network update
    target_update_freq - number of policy network update steps between each
                         target network update
    mini_batch         - size of the minibatch
    discount           - discount to be used for the problem
    replay_mem         - size of the replay buffer
    lr                 - learning rate or dict containing LR scheduler
                         arguments
    eps                - eps value or dict containing 'start', 'end'
                         and 'period' keys for scheduling
    pri_buf_args       - dict containing 'alpha', 'beta' and 'period'
                         (beta annealing) keys for prioritized replay buffer
    distrib_args       - dict containg 'atoms', 'min_val' (minimum value) and
                         'max_val' (maximum value) keys for the distributional
                         value network
    clip_grad          - value for clipping gradient norms
    learn_start        - time step after which to start learning
    check_pts          - episodes at which the policy network state dict
                         should be saved
    save_path          - path to save the neural network parameters
    load_path          - load model state dict from the provided path
                         (optimizer and other data not restored)
    no_duel            - flag to turn of duelling networks
    no_double          - flag to turn off double Q-networks
    no_priority        - flag to turn off prioritized buffers
    no_noise           - flag to turn off noisy layers
    no_distrib         - flag to turn off distributional value learning

    """

    def __init__(self, num_actions, n_net, seed=None, n_step=1,
                 policy_update_freq=2, target_update_freq=75,
                 mini_batch=32, discount=0.999, replay_mem=10000,
                 lr={'start': 0.0005, 'end': 0.0001, 'period': 500},
                 eps={'start': 0.9, 'end': 0.05, 'period': 2500},
                 pri_buf_args={'alpha': 0.7, 'beta': (0.5, 1), 'period': 1e6},
                 distrib_args={'atoms': 21}, clip_grads=10, learn_start=None,
                 check_pts=[], save_path=None, load_path=None, use_gpu=True,
                 no_duel=False, no_double=False, no_priority_buf=False,
                 no_noise=False, no_distrib=False):
        self.num_actions = num_actions
        self.n_step = n_step
        self.policy_update_freq = policy_update_freq
        self.target_update_freq = target_update_freq * policy_update_freq
        self.mini_batch = mini_batch
        self.discount = discount
        self.replay_mem = replay_mem
        self.clip_grads = clip_grads
        self.learn_start = learn_start
        self.load_path = load_path
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
        if torch.cuda.is_available() and use_gpu:
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
            self.delta_z =\
                (self.max_val - self.min_val)/(self.num_atoms - 1)
            self.z_atoms = [self.min_val]
            for _ in range(1, self.num_atoms):
                self.z_atoms.append(self.z_atoms[-1] + self.delta_z)
            self.z_atoms = torch.tensor(self.z_atoms, device=self.device,
                                        dtype=torch.float32)

            def network(act): return n_net(act, self.z_atoms)
        else:
            network = n_net
        # setup the neural network
        self.policy_net = network(self.num_actions)
        self.target_net = network(self.num_actions)
        if self.load_path is not None:
            self.policy_net.load_state_dict(torch.load(self.load_path))
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
        if self.learn_start is None:
            self.learn_start = self.replay_mem
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
        # set up save points
        if save_path is not None and len(check_pts) >= 0:

            self.save_path = save_path + '/' + \
                             datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            os.makedirs(self.save_path)
            self.check_pts = check_pts
            self.save_model = True
        else:
            self.save_model = False

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
        values = values.squeeze()
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
                len(self.replay) >= self.learn_start):
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
                                       np.zeros_like(self.obs_buffer[-i], dtype='float32'),
                                       n_reward, False, np.inf))
        # clear stored memory
        self.reward_buffer.clear()
        self.obs_buffer.clear()
        self.action_buffer.clear()
        # update policy network
        if (self.steps % self.policy_update_freq == 0 and
                len(self.replay) >= self.learn_start):
            self._compute_loss()
            self._optimize()
            # update target network
            if self.steps % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        # save model
        if self.save_model and self.episodes in self.check_pts:
            state_dict = self.policy_net.state_dict()
            for key, val in state_dict.items():
                state_dict[key] = val.cpu()
            torch.save(state_dict, self.save_path + '/' + str(self.episodes))
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
                               dtype=torch.float32)
        next_obs = torch.tensor(next_obs, device=self.device,
                                dtype=torch.float32)[t_masks]
        t_mask_tensor = torch.tensor(t_masks, device=self.device,
                                     dtype=torch.float32)
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
            target = rewards.unsqueeze(-1) + \
                torch.ger(t_mask_tensor, self.n_disc*self.z_atoms)
            target.clamp_(self.min_val, self.max_val)
            # project non-terminal state targets on to the support vectors
            z_expan = self.z_atoms.view(1, -1, 1)
            target_expan = target.unsqueeze(1)
            abs_diff = torch.abs(target_expan - z_expan)/self.delta_z
            proj_coeff = torch.clamp(1 - abs_diff, 0, 1)
            # calulate next state-action value probability
            nxt_obs_no = [i for i in range(next_obs.shape[0])]
            next_probs = torch.ones((self.mini_batch, self.num_atoms),
                                    device=self.device, dtype=torch.float32)/self.num_atoms
            next_probs[t_masks] = self.target_net(next_obs)[nxt_obs_no, max_actions, :]
            next_probs.unsqueeze_(1)
            # Distribute the probability of non-terminal targets
            target_probs = (next_probs * proj_coeff).sum(-1)

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
