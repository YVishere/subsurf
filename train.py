import torch
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import torch.nn as nn
from itertools import count

from agent import ReplayMemory, Transition
from model import DQCNN
from game_env import SubwayEnv
from scripts import start_game

from enum import Enum
import time

import keyboard 

# Add this function before your training loop
def check_for_exit_key():
    """Check if 'C' key is pressed and exit if so"""
    if keyboard.is_pressed('c'):
        print("\nUser interrupted training with 'C' key. Exiting...")
        # Clean up any resources
        plt.close('all')
        try:
            env.close()  # Close environment if it has a close method
        except:
            pass
        import sys
        sys.exit(0)  # Exit program

print("Starting...")

class Action(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    NONE = 4
    
actions = [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN, Action.NONE]

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.95  # Increase from 0.9
EPS_END = 0.1     # Increase from 0.05
EPS_DECAY = 0.2   # Slower decay (increase from 0.05)
TAU = 0.005
LR = 1e-4

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

n_actions = len(actions)

first_img = Image.open("bluestacks_screenshot.png")
np_img = np.array(first_img)
gray_img = np.dot(np_img[..., :3], [0.2989, 0.5870, 0.1140])

env = SubwayEnv(frame_stack=1, frame_size=gray_img.shape)

gray_img = torch.unsqueeze(torch.tensor(gray_img), axis=0)  # Add channel dimension
n_obs = gray_img.shape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

policy_net = DQCNN(n_actions, n_obs).to(device)
target_net = DQCNN(n_actions, n_obs).to(device)
target_net.load_state_dict(policy_net.state_dict())

# Add to your model definition:
dropout_rate = 0.2  # Add dropout to FC layers
weight_decay = 1e-5  # Add to optimizer
optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR, weight_decay=weight_decay, amsgrad=True)

memory = ReplayMemory(10000)

steps_done = 0

print("Initialized")

def select_action(state):
    global steps_done
    sample = np.random.random()
    
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1,1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)

    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        means = durations_t.unfold(0,100,1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    
    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

###Training loop

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Safety check for tensor shapes
    sample_state = batch.state[0]
    expected_shape_len = len(sample_state.shape)
    
    # Skip any non-matching tensors
    valid_states = []
    valid_actions = []
    valid_next_states = []
    valid_rewards = []
    
    for i in range(len(batch.state)):
        if batch.next_state[i] is not None and len(batch.next_state[i].shape) == expected_shape_len:
            valid_states.append(batch.state[i])
            valid_actions.append(batch.action[i])
            valid_next_states.append(batch.next_state[i])
            valid_rewards.append(batch.reward[i])
    
    # Skip batch if not enough valid samples
    if len(valid_states) < 10:
        print(f"Not enough valid samples: {len(valid_states)}")
        return
    
    # Continue with valid samples only
    state_batch = torch.cat(valid_states)
    action_batch = torch.cat(valid_actions)
    reward_batch = torch.cat(valid_rewards)
    
    non_final_mask = torch.ones(len(valid_next_states), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat(valid_next_states)
    
    # Get predicted Q-values
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Create tensor with correct size - THIS IS THE KEY FIX
    next_state_values = torch.zeros(len(valid_states), device=device)
    
    # Debug shape checking
    if len(non_final_next_states.shape) < 4:  # Should be [batch, channels, height, width]
        print(f"Problem with tensor shape: {non_final_next_states.shape}")
        return

    # Get target Q-values
    with torch.no_grad():
        next_state_values = target_net(non_final_next_states).max(1).values

    # Calculate expected Q-values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()

    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# 1. Use torch.jit to compile your model
policy_net_optimized = torch.jit.script(policy_net)
target_net_optimized = torch.jit.script(target_net)

if torch.cuda.is_available():
    num_eps = 600
else:
    num_eps = 50

print("Starting game...")
print("Training...")
print("-----------------------")

for episode in range(num_eps):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    total_reward = 0
    
    for t in count():
        action = select_action(state)
        obs, reward, done, _, info = env.step(action.item())
        
        # Reshape tensor properly based on observation shape
        if isinstance(obs, np.ndarray):
            if len(obs.shape) == 2:  # Single grayscale frame [H,W]
                next_state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            elif len(obs.shape) == 3:  # Frame stack [stack,H,W]
                next_state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            else:
                print(f"Unexpected observation shape: {obs.shape}")
                next_state = state  # Reuse previous state as fallback
        else:
            print(f"Unexpected observation type: {type(obs)}")
            next_state = state

        # Check for exit key after each action
        check_for_exit_key()  # ADD THIS LINE
        
        reward = torch.tensor([reward], device=device)
        total_reward += reward.item()

        memory.push(state, action, next_state, reward, done)

        state = next_state

        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            # plot_durations()
            print("============")
            print(f"Episode {episode}, Score: {info['score']}, Reward: {total_reward:.2f}")
            print("============")
            break