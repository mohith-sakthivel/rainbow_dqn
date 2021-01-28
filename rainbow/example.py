import gym
import time
import torch
import os
import numpy as np

import tqdm.autonotebook as auto
import matplotlib.pyplot as plt

from rainbow.rainbow import Rainbow
from rainbow.models import NoisyDistNet, ConvDQN
from rainbow.utils import choose_max, plot_var_history, get_model_name, preprocess_binary
from torch.utils.tensorboard import SummaryWriter


def make_gym_env(Environment):
    def make_env():
        return gym.make(Environment)
    return make_env


def best_check_point(make_env, act_freq, network, process_obs,
                     check_pts, save_path, atoms=None, trials=10):
    env = make_env()
    if atoms is None:
        model = network(env.action_space.n)
    else:
        model = network(env.action_space.n, atoms)
    avg_returns = []
    for check_pt in check_pts:
        state_dict = torch.load(save_path + '/' + str(check_pt))
        model.load_state_dict(state_dict)
        avg_returns.append([])
        for i in range(1, trials+1, 1):
            env.seed(i)
            obs = env.reset()
            obs_stack = [obs]
            done = False
            time_step = 0
            total_reward = 0
            while not done:
                if len(obs_stack) == act_freq or time_step == 0:
                    act = choose_max(model.get_values(process_obs(obs_stack)))
                    obs_stack = []
                obs, reward, done, info = env.step(act)
                obs_stack.append(obs)
                total_reward += reward
                time_step += 1
            avg_returns[-1].append(total_reward)
    env.close()
    avg_returns = np.array(avg_returns).sum(axis=-1)
    max_idx = np.argmax(avg_returns)
    # get best performing model
    state_dict = torch.load(save_path + '/' + str(check_pts[max_idx]))
    model.load_state_dict(state_dict)
    print('Best performing model: Episode %d' % (check_pts[max_idx]))
    for check_pt in check_pts:
        os.remove(save_path + '/' + str(check_pt))
    torch.save(model.state_dict(), save_path + '/' + 'model_state_dict')
    return model


def view_agent(make_env, act_freq, network, process_obs,
               save_path, atoms=None, trials=1):
    env = make_env()
    if atoms is None:
        model = network(env.action_space.n)
    else:
        model = network(env.action_space.n, atoms)

    state_dict = torch.load(save_path + '/' + 'model_state_dict')
    model.load_state_dict(state_dict)
    trial_returns = []
    for i in range(1, trials+1, 1):
        print('Trial: %d' % (i))
        env.seed(i)
        obs = env.reset()
        obs_stack = [obs]
        done = False
        time_step = 0
        total_reward = 0
        while not done:
            env.render()
            time.sleep(0.01)
            if len(obs_stack) == act_freq or time_step == 0:
                act = choose_max(model.get_values(process_obs(obs_stack)))
                obs_stack = []
            obs, reward, done, info = env.step(act)
            obs_stack.append(obs)
            total_reward += reward
            time_step += 1
        print('Reward: %d' % total_reward)
        trial_returns.append(total_reward)
    env.close()


def run_experiment(make_env, process_obs, act_freq,
                   num_runs, num_episodes, agent_args,
                   tb_path=None, watch_agent=True):
    reward_history = []
    agents_hist = []
    env = make_env()
    assert isinstance(env.action_space, gym.spaces.Discrete), \
        "Action space is not discrete"
    act_dim = env.action_space.n
    for i, agent_kwargs in enumerate(agent_args):
        reward_history.append([])
        # Start the runs for each setting
        for run in range(1, num_runs+1):
            reward_history[i].append([])
            env.seed(run)
            np.random.seed(run)
            agent_kwargs["seed"] = run
            agent = Rainbow(act_dim, **agent_kwargs)
            if tb_path:
                writer = SummaryWriter(get_model_name(agent_kwargs, tb_path))
            # Start the episodes
            for episode in auto.tqdm(range(1, num_episodes+1),
                                     desc='Config %d | Run %d ' % (i+1, run),
                                     leave=False):
                observation = env.reset()
                if process_obs is not None:
                    observation = process_obs([observation])
                done = False
                time_step = 0
                noise = 0
                action = agent.start(observation)
                interim_reward = 0
                interim_obs = []
                # Start interaction with environment
                while not done:
                    # Take a step in the environment
                    observation, reward, done, info = env.step(action)
                    interim_reward += reward
                    interim_obs.append(observation)
                    time_step += 1
                    if not done and time_step % (act_freq+noise) == 0:
                        # Get next action from agent
                        interim_obs = interim_obs[-act_freq:]
                        if process_obs is not None:
                            interim_obs = process_obs(interim_obs)
                        action = agent.take_step(interim_reward, interim_obs)
                        noise = np.random.randint(0, 3)
                        interim_reward = 0
                        interim_obs = []
                    elif done:
                        episode_reward = agent.end(reward)
                # post episode processing
                reward_history[i][run-1].append(episode_reward)
                if tb_path:
                    writer.add_scalar('Episode Reward',
                                      episode_reward, episode)
                    if episode % 25 == 0:
                        data = agent.get_train_data()
                        if data is not None:
                            writer.add_scalar('Loss', data.get('loss'), episode)
            # Find the best performing checkpoint
            agents_hist.append(best_check_point(make_env, act_freq,
                                                agent_kwargs['n_net'], process_obs,
                                                agent_kwargs['check_pts'],
                                                agent.save_path,
                                                agent.z_atoms.cpu()))
            # View the agent's performance
            if watch_agent:
                view_agent(make_env, act_freq, agent_kwargs['n_net'],
                           process_obs, agent.save_path,
                           agent.z_atoms.cpu())
            if tb_path:
                writer.close()
    env.close()
    return reward_history, agents_hist


def run_pong(runs=1, episodes=10000, render=True):
    # Setup pong environment
    Environment = 'Pong-v0'
    test_env = make_gym_env(Environment)

    ag_args = [
               {'n_step': 3,
                'n_net': lambda act, atoms: ConvDQN(4, act, atoms),
                'policy_update_freq': 4, 'target_update_freq': 1250,
                'mini_batch': 32, 'discount': 0.99, 'replay_mem': 250000,
                'lr': {'start': 5e-4, 'end': 2.5e-4, 'period': 10000},
                'eps': 0,
                'pri_buf_args': {'alpha': 0.7, 'beta': (0.5, 1), 'period': 1e6},
                'distrib_args': {'atoms': 21, 'min_val': -25, 'max_val': 25},
                'clip_grads': 20, 'learn_start': 1e4,
                'check_pts': [i*1000 for i in range(1, 100, 1)],
                'save_path': 'data/Pong-v0',
                'no_duel': False, 'no_double': False,
                'no_priority_buf': False, 'no_noise': False,
                'no_distrib': False},
                ]

    return run_experiment(make_env=test_env, act_freq=4,
                          process_obs=lambda x: preprocess_binary(x, 4, True),
                          num_runs=runs, num_episodes=episodes,
                          agent_args=ag_args,
                          tb_path='runs/Pong-v0', watch_agent=render)


def run_cartpole(runs=1, episodes=250, render=True):
    # Setup cartpole environment
    Environment = 'CartPole-v1'
    test_env = make_gym_env(Environment)

    ag_args = [
               {'n_step': 3,
                'n_net': lambda act, atoms: NoisyDistNet(4, [64, 128, 64], act, atoms),
                'policy_update_freq': 2, 'target_update_freq': 75,
                'mini_batch': 32, 'discount': 0.999, 'replay_mem': 10000,
                'lr': {'start': 1e-3, 'end': 1e-4, 'period': 5000},
                'eps': 0, 'learn_start': 1e2,
                'pri_buf_args': {'alpha': 0.7, 'beta': (0.5, 1), 'period': 1e6},
                'distrib_args': {'atoms': 21, 'min_val': 0, 'max_val': 500},
                'clip_grads': None,
                'check_pts': [150, 200, 250], 'save_path': 'data/CartPole-v1',
                'no_duel': False, 'no_double': False, 'no_priority_buf': False,
                'no_noise': False, 'no_distrib': False},
               ]

    return run_experiment(make_env=test_env, act_freq=1,
                          process_obs=lambda x: x[-1],
                          num_runs=runs, num_episodes=episodes,
                          agent_args=ag_args,
                          tb_path='runs/CartPole-v1', watch_agent=render)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        help='Environment instantiation method')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number runs to train')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of episodes to train')
    parser.add_argument('--render', action="store_true",
                        help='Render agent performance after training')
    args = parser.parse_args()
    run_success = True

    # Collect experiment arguments passed
    exp_args = {}
    exp_args['render'] = args.render
    exp_args['runs'] = args.runs
    if args.episodes is not None:
        exp_args['episodes'] = args.episodes

    if args.env in ['cartpole', 'CartPole', 'CartPole-v1']:
        print("Running CartPole-v1 environment")
        reward_hist, agent_hist = run_cartpole(**exp_args)
    elif args.env in ['pong', 'Pong', 'Pong-v0']:
        print('Running Pong-v0 environment')
        reward_hist, agent_hist = run_pong(**exp_args)
    else:
        print('invalid argument')
        run_success = False

    # Plot Results
    if run_success:
        plot_args = {'x_label': 'Episode',
                     'y_label': 'Reward during Episode (Average)',
                     'log_scale': False}
        labels = ["Rainbow DQN"]

        plot_var_history(reward_hist, labels, **plot_args)
        plt.show()
