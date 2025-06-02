import torch
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import torch.nn as nn
from itertools import count
import gc

from agent import ReplayMemory, Transition
from model import DQCNN
from game_env import SubwayEnv
from scripts import start_game

from enum import Enum
import time

import keyboard 

frame_stack = 15  # Number of frames to stack
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

BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9  # Increase from 0.9
EPS_END = 0.1     # Increase minimum exploration
EPS_DECAY = 0.2   # Slower decay rate
TAU = 0.001
LR = 1e-4

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

n_actions = len(actions)

first_img = Image.open("./reference_pics/bluestacks_screenshot.png")
np_img = np.array(first_img)
gray_img = np.dot(np_img[..., :3], [0.2989, 0.5870, 0.1140])

env = SubwayEnv(frame_stack=frame_stack, frame_size=(84, 84))  # Standard RL size

gray_img = torch.unsqueeze(torch.tensor(gray_img), axis=0)  # Add channel dimension
n_obs = (frame_stack, gray_img.shape[1], gray_img.shape[2])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

policy_net = DQCNN(n_actions, n_obs).to(device)
target_net = DQCNN(n_actions, n_obs).to(device)
target_net.load_state_dict(policy_net.state_dict())

# Add to your model definition:
dropout_rate = 0.2  # Add dropout to FC layers
weight_decay = 1e-5  # Add to optimizer
optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR, weight_decay=weight_decay, amsgrad=True)

memory = ReplayMemory(20000)

steps_done = 0

print("Initialized")

# Add to your select_action function
def select_action(state):
    global steps_done, episode
    # Force exploration every 10 episodes
    if episode % 10 == 0:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    
    # Make sure state is float32 before passing to model
    if state.dtype == torch.uint8:
        state = state.to(torch.float32) / 255.0
    
    global steps_done
    sample = np.random.random()
    
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            q_values = policy_net(state)
            
            # Gradually decreasing bias for NONE action
            none_bias = 0.3 * max(0, 1 - steps_done/30000)  # Less bias, shorter duration
            q_values[0, 4] += none_bias  # NONE action bias
            
            return q_values.max(1)[1].view(1, 1)
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
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                 device=device, dtype=torch.bool)
    
    # Skip if no valid next states
    if non_final_mask.sum() == 0:
        return
    
    # Explicitly convert state_batch to float32 BEFORE sending to GPU
    state_batch = torch.cat(batch.state)
    # Fix the type conversion
    state_batch = state_batch.to(torch.float32).to(device) / 255.0
    
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # Only use non-final next states with correct type conversion
    non_final_next_states_list = [s for s in batch.next_state if s is not None]
    if len(non_final_next_states_list) > 0:
        non_final_next_states = torch.cat(non_final_next_states_list)
        # Fix the type conversion here too
        non_final_next_states = non_final_next_states.to(torch.float32).to(device) / 255.0
    else:
        non_final_next_states = None  # Should not happen due to earlier check
    
    # Move non_final_next_states to the correct device
    if non_final_next_states is not None:
        non_final_next_states = non_final_next_states.to(device)

    # Continue with regular DQN update
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# 1. Use torch.jit to compile your model
# policy_net_optimized = torch.jit.script(policy_net)
# target_net_optimized = torch.jit.script(target_net)

if torch.cuda.is_available():
    num_eps = 1200
else:
    num_eps = 50

print("Starting game...")
print("Training...")
print("-----------------------")

# Modified memory.push() preprocessing
def process_for_memory(tensor):
    if tensor is None:
        return None
    # Scale to 0-255 and convert to uint8
    return (tensor.detach().cpu() * 255).to(torch.uint8)

for episode in range(num_eps):
    with torch.no_grad():
        state, _ = env.reset()
        if len(state.shape) == 2:  # Single frame [H, W]
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        elif len(state.shape) == 3:  # Frame stack [stack, H, W]
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            print(f"Warning: Unexpected state shape: {state.shape}")
            # Try to fix it
            state = state.reshape(state.shape[-2], state.shape[-1])  # Take last two dimensions
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    total_reward = 0
    
    for t in count():
        check_for_exit_key()  # Check for exit key at each step
        action = select_action(state)
        obs, reward, done, _, info = env.step(action.item())
        
        # Process observations with no_grad
        with torch.no_grad():
            # Replace your current tensor conversion code with this:
            if isinstance(obs, np.ndarray):
                # Check the shape and properly format it for your CNN
                if len(obs.shape) == 2:  # Single frame [H, W]
                    # Add batch and channel dimensions: [1, 1, H, W]
                    next_state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                elif len(obs.shape) == 3:  # Already stacked [stack, H, W]
                    # Just add batch dimension: [1, stack, H, W]
                    next_state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                else:
                    print(f"Warning: Unexpected observation shape: {obs.shape}")
                    # Try to fix the shape
                    obs = obs.reshape(obs.shape[-2], obs.shape[-1])  # Take last two dimensions
                    next_state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            else:
                print(f"Warning: Unexpected observation type: {type(obs)}")
                next_state = None

            # Only store in memory if state is valid
            if next_state is not None:
                reward = torch.tensor([reward], device=device)
                total_reward += reward.item()

                # Ensure all are detached and on CPU, and log info
                s = state.detach().cpu() if isinstance(state, torch.Tensor) else state
                a = action.detach().cpu() if isinstance(action, torch.Tensor) else action
                r = reward.detach().cpu() if isinstance(reward, torch.Tensor) else reward
                ns = next_state.detach().cpu() if next_state is not None and isinstance(next_state, torch.Tensor) else next_state
                d = done
                # During storage
                s = process_for_memory(state)
                ns = process_for_memory(next_state)
                memory.push(s, a, r, ns, d)

                state = next_state

        optimize_model()

        # Move the soft update to CPU to save GPU memory
        with torch.no_grad():
            # Get dictionaries
            target_net.cpu()
            policy_net_state_dict = {k: v.cpu() for k, v in policy_net.state_dict().items()}
            target_net_state_dict = target_net.state_dict()
            
            # Update on CPU
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            
            # Load updated weights
            target_net.load_state_dict(target_net_state_dict)
            
            # Clean up to free memory
            del policy_net_state_dict
            del target_net_state_dict
            
            # Move target_net back to the appropriate device
            target_net.to(device)

        if done:
            episode_durations.append(t + 1)
            # plot_durations()
            print("============")
            print(f"Episode {episode}, Score: {info['score']}, Reward: {total_reward:.2f}")
            print("============")
            break

    # After optimize_model
    if t % 10 == 0:  # More frequent cleanup
        gc.collect()
        torch.cuda.empty_cache()
