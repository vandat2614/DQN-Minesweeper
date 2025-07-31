import torch
import time
from .utils import convert_state_to_onehot

def test(env, model, device, num_episodes=100):
    model = model.to(device)
    model.eval()

    for episode in range(num_episodes):
        state, info = env.reset()
        while True:

            with torch.no_grad():
                state_tensor = convert_state_to_onehot(state).unsqueeze(0).to(device)
                q_values = model(state_tensor).clone()
                mask = torch.tensor(env.valid_actions, dtype=torch.bool, device=device)  

                q_values = q_values.masked_fill(~mask.unsqueeze(0), -1e9)
                action = q_values.argmax(dim=1).item()

            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state

            time.sleep(0.5)

            if terminated or truncated:

                if info['is_win']:
                    print(f'Win in episode {episode+1}')
                break