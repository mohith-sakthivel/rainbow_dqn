import gym
import time

from rainbow import Rainbow
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as T
import torch
from PIL import Image
import numpy as np
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(env, screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(env, screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).numpy()


def run_experient(env, num_runs, num_episodes, agent_args,
                  tb_path=None, model_name='model_joe',
                  render_env=False, plot_value_func=False,
                  plot_state_visit=False):
    reward_history = []
    agents_hist = []
    # state_dim = env.observation_space.shape[0]
    env.reset()
    init_screen = get_screen(env)
    _, _, screen_height, screen_width = init_screen.shape
    state_dim = (screen_height, screen_width)
    assert isinstance(env.action_space, gym.spaces.Discrete), \
        "Action space is not discrete"
    act_dim = env.action_space.n
    for i, (Agent, agent_kwargs) in enumerate(agent_args):
        reward_history.append([])
        # Start the runs for each setting
        for run in range(1, num_runs+1):
            reward_history[i].append([])
            env.seed(run)
            agent_kwargs["seed"] = run
            agent = Agent(state_dim, act_dim, **agent_kwargs)
            if run == 1 and tb_path:
                writer = SummaryWriter(tb_path + '/' + agent.__str__() +
                                       '/' + model_name)
            # Start the episodes
            for episode in range(1, num_episodes+1):
                _ = env.reset()
                done = False
                time_step = 0

                last_screen = get_screen(env)
                current_screen = get_screen(env)
                observation = current_screen - last_screen

                action = agent.start(observation)
                # Start interaction with environment
                while not done:
                    if render_env:
                        env.render()
                        time.sleep(0.001)
                    # Take a step in the environment
                    _, reward, done, info = env.step(action)
                    time_step += 1
                    last_screen = current_screen
                    current_screen = get_screen(env)
                    observation = current_screen - last_screen
                    if not done:
                        # Get next action from agent
                        action = agent.take_step(reward, observation)
                    else:
                        episode_reward = agent.end(reward)
                # post episode processing
                reward_history[i][run-1].append(episode_reward)
                if run == 1 and tb_path:
                    writer.add_scalar('Episode Reward',
                                      episode_reward, episode)
                    if episode % 10 == 0:
                        data = agent.get_train_data()
                        writer.add_scalar('Loss', data.get('loss'), episode)
            if run == 1:
                agents_hist.append(agent)
                if tb_path:
                    writer.close()
    env.close()
    return reward_history, agents_hist


# Setup Cart Pole environment
Environment = 'CartPole-v1'
cart_pole_env = gym.make(Environment)

agent_args = [(Rainbow, {})]

reward_hist, agent_hist = run_experient(cart_pole_env, 1, 10000, agent_args,
                                        tb_path='runs', render_env=False)
