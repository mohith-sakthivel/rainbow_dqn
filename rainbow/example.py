import gym
import time

import tqdm.autonotebook as auto
import matplotlib.pyplot as plt

from rainbow import Rainbow
from models import NoisyDistNet, ConvDQN
from utils import plot_var_history, get_model_name, preprocess
from torch.utils.tensorboard import SummaryWriter


def run_experient(env, process_obs, act_freq,
                  num_runs, num_episodes, agent_args,
                  tb_path=None,
                  render_env=False, plot_value_func=False,
                  plot_state_visit=False):
    reward_history = []
    agents_hist = []
    assert isinstance(env.action_space, gym.spaces.Discrete), \
        "Action space is not discrete"
    act_dim = env.action_space.n
    for i, agent_kwargs in enumerate(agent_args):
        reward_history.append([])
        # Start the runs for each setting
        for run in range(1, num_runs+1):
            reward_history[i].append([])
            env.seed(run)
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
                action = agent.start(observation)
                interim_reward = 0
                interim_obs = []
                # Start interaction with environment
                while not done:
                    if render_env:
                        env.render()
                        time.sleep(0.001)
                    # Take a step in the environment
                    observation, reward, done, info = env.step(action)
                    interim_reward += reward
                    interim_obs.append(observation)
                    time_step += 1
                    if not done and time_step % act_freq == 0:
                        # Get next action from agent
                        if process_obs is not None:
                            interim_obs = process_obs(interim_obs)
                        action = agent.take_step(interim_reward, interim_obs)
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
            if run == 1:
                agents_hist.append(agent)
            if tb_path:
                writer.close()
    env.close()
    return reward_history, agents_hist


def run_pong(runs=1, episodes=1000):
    # Setup pong environment
    Environment = 'Pong-v0'
    test_env = gym.make(Environment)

    ag_args = [
               {'n_step': 3,
                'n_net': lambda act, atoms: ConvDQN(4, act, atoms),
                'policy_update_freq': 4, 'target_update_freq': 5000,
                'mini_batch': 32, 'discount': 0.99, 'replay_mem': 100000,
                'lr': {'start': 5e-4, 'end': 1e-4, 'period': 5000},
                'eps': 0,
                'pri_buf_args': {'alpha': 0.7, 'beta': (0.5, 1), 'period': 1e6},
                'distrib_args': {'atoms': 21, 'min_val': -25, 'max_val': 25},
                'clip_grads': 20, 'learn_start': 1e5,
                'check_pts': [250, 500, 750], 'save_path': 'data/pong',
                'no_duel': False, 'no_double': False,
                'no_priority_buf': False, 'no_noise': False,
                'no_distrib': False},
                ]

    return run_experient(env=test_env, act_freq=4,
                         process_obs=lambda x: preprocess(x, 4),
                         num_runs=runs, num_episodes=episodes,
                         agent_args=ag_args,
                         tb_path='runs/pong', render_env=False)


def run_cartpole(runs=1, episodes=250):
    # Setup cartpole environment
    Environment = 'CartPole-v1'
    test_env = gym.make(Environment)

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
                'check_pts': [100, 200, 250], 'save_path': 'data/cartpole',
                'no_duel': False, 'no_double': False, 'no_priority_buf': False,
                'no_noise': False, 'no_distrib': False},
               ]

    return run_experient(env=test_env, act_freq=1,
                         process_obs=lambda x: x[-1],
                         num_runs=runs, num_episodes=episodes,
                         agent_args=ag_args,
                         tb_path='runs/cartpole', render_env=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--episodes', type=int, default=250)
    args = parser.parse_args()
    run_success = True

    if args.env in ['cartpole', 'CartPole', 'CartPole-v1']:
        print("Running CartPole-v1 environment")
        reward_hist, agent_hist = run_cartpole(args.runs, args.episodes)
    elif args.env in ['pong', 'Pong', 'Pong-v0']:
        print('Running Pong-v0 environment')
        reward_hist, agent_hist = run_pong(args.runs, args.episodes)
    else:
        print('invalid argument')
        run_success = False

    # Plot Results
    if run_success:
        plot_args = {'x_label': 'Episode',
                     'y_label': 'Reward during Episode (Average)',
                     'log_scale': False,
                     'y_ticks': [25, 50, 100, 200, 300, 400, 500]}
        labels = ["Rainbow DQN"]

        plot_var_history(reward_hist, labels, **plot_args)
        plt.show()
