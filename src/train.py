import random
import torch
import torch.nn as nn

import os
import json
import copy
import itertools
from datetime import datetime

from .experience_replay import ExperienceReplay
from .utils import convert_state_to_onehot

def update(model: nn.Module, batch: tuple, optimizer: torch.optim.Optimizer, 
                criterion: nn.Module, gamma: float, device: torch.device, target_network: nn.Module = None):
    
    batch = list(zip(*batch))
    states, actions, rewards, next_states, dones = batch

    states = torch.stack(states).float().to(device)
    next_states = torch.stack(next_states).float().to(device)

    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    current_q_values = model(states).gather(dim=1, index=actions.unsqueeze(1))
    
    eval_model = target_network if target_network is not None else model
    next_q_values = eval_model(next_states).max(dim=1)[0].detach()
    target_q_values = rewards + gamma * next_q_values * (1 - dones)
    
    loss = criterion(current_q_values.squeeze(), target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train(env, model, config, device):
    model = model.to(device)

    if config['target_network']:
        target_network = copy.deepcopy(model)
        target_network.load_state_dict(model.state_dict())
        target_network = target_network.to(device)
        target_network.eval()
    else:
        target_network = None

    num_episodes = config.get('num_episodes', None)
    batch_size = config['batch_size']
    gamma = config['gamma']
    epsilon_max = config['epsilon_max']
    epsilon_min = config['epsilon_min']
    epsilon_decay = config['epsilon_decay']
    learning_rate = config['learning_rate']
    memory_size = config['memory_size']
    target_update_freq = config.get('target_update_freq', None)
    
    base_dir = config['base_dir']
    weights_dir = os.path.join(base_dir, "train", "weights")
    log_path = os.path.join(base_dir, "train", "log")
    train_log_path = os.path.join(base_dir, "train", "train_log.json")
    os.makedirs(weights_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    memory = ExperienceReplay(memory_size)

    epsilon = epsilon_max
    best_reward = float('-inf')

    train_log = []

    episode_iter = itertools.count() if num_episodes is None else range(num_episodes)
    for episode in episode_iter:
        state, info = env.reset()
        total_reward = 0
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        loss = None

        while True:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = convert_state_to_onehot(state).unsqueeze(0).to(device)
                    q_values = model(state_tensor)
                    mask = torch.tensor(env.valid_actions, dtype=torch.bool, device=device)  
                    q_values[0][~mask] = -1e9
                    action = q_values.argmax(dim=1).item()

            next_state, reward, terminated, truncated, info = env.step(action)

            state_onehot = convert_state_to_onehot(state)         
            next_state_onehot = convert_state_to_onehot(next_state)

            memory.push((state_onehot, action, reward, next_state_onehot, terminated))

            total_reward += reward
            state = next_state

            if len(memory) >= batch_size:
                batch = memory.sample(batch_size)
                loss = update(model, batch, optimizer, criterion, gamma, device, target_network)
            
            if terminated or truncated:
                break
        is_win = info['is_win']

        train_log.append({
            "episode": episode + 1,
            "reward": total_reward,
            "is_win": is_win,
            "loss": loss,
            "epsilon": epsilon
        })

        if config['target_network'] and (episode + 1) % target_update_freq == 0:
            target_network.load_state_dict(model.state_dict())
        
        torch.save(model.state_dict(), f'{weights_dir}/last.pt')


        if info["is_win"]:
            win_path = os.path.join(weights_dir, f"win_ep{episode + 1}.pt")
            torch.save(model.state_dict(), win_path)
            
            date_hour = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"{date_hour}: Win recorded: episode {episode + 1}, reward {total_reward:.2f}"
            print(log_message)
            with open(log_path, 'a') as file:
                file.write(log_message + '\n')

        if total_reward > best_reward:
            best_reward = total_reward

            date_hour = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"{date_hour}: New best reward: episode {episode + 1}, reward {total_reward:.2f}, status {'win' if info['is_win'] else 'loss'}"
            
            print(log_message)
            with open(log_path, 'a') as file:
                file.write(log_message + '\n')

            # Lưu model tốt nhất vào best.pt
            torch.save(model.state_dict(), os.path.join(weights_dir, "best.pt"))

    with open(train_log_path, "w") as file:
        json.dump(train_log, file, indent=4)
