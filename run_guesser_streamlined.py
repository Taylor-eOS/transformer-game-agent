import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from utils_guesser import print_memory

class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GRID_SIZE = 3
    TEST_MODE = False
    EVALUATION = True
    EPOCHS = 2 if TEST_MODE else 8192
    TRAIN_STEPS = 2 if TEST_MODE else 128
    BATCH_SIZE = 2 if TEST_MODE else 32
    SEQUENCE_LENGTH = 5
    LEARNING_RATE = 0.001
    MAX_AGENT_STEPS = 10
    VAL_TESTS = 100
    VALIDATION_STEPS = 10
    EMBED_SIZE = 32
    NUM_HEADS = 2
    NUM_LAYERS = 3

class TransformerModel(nn.Module):
    def __init__(self, feature_length=3, seq=Config.SEQUENCE_LENGTH * 6, num_actions=4):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(feature_length, Config.EMBED_SIZE)
        encoder_layer = nn.TransformerEncoderLayer(d_model=Config.EMBED_SIZE, nhead=Config.NUM_HEADS, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=Config.NUM_LAYERS)
        self.fc = nn.Linear(Config.EMBED_SIZE, num_actions)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

class GridEnv:
    def __init__(self):
        self.grid_size = Config.GRID_SIZE
        self.reset()

    def normalize_position(self, pos):
        return pos[0] / (self.grid_size - 1), pos[1] / (self.grid_size - 1)

    def get_state(self, position):
        state = []
        norm_x, norm_y = self.normalize_position(self.agent_pos)
        state.append([position, 0, norm_x])
        state.append([position, 0, norm_y])
        norm_x, norm_y = self.normalize_position(self.food_pos)
        state.append([position, 1, norm_x])
        state.append([position, 1, norm_y])
        norm_x, norm_y = self.normalize_position(self.poison_pos)
        state.append([position, 2, norm_x])
        state.append([position, 2, norm_y])
        return np.array(state, dtype=np.float32)

    def reset(self):
        positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        self.agent_pos = random.choice(positions)
        positions.remove(self.agent_pos)
        self.food_pos = random.choice(positions)
        positions.remove(self.food_pos)
        self.poison_pos = random.choice(positions)
        self.previous_pos = None
        initial_states = [self.get_state(0) for _ in range(Config.SEQUENCE_LENGTH)]
        flat_initial_states = [vec for state in initial_states for vec in state]
        return deque(flat_initial_states, maxlen=Config.SEQUENCE_LENGTH * 6)

    def step(self, action, position):
        self.previous_pos = self.agent_pos
        new_row, new_col = self.agent_pos
        if action == 0:
            new_row = max(new_row - 1, 0)
        elif action == 1:
            new_row = min(new_row + 1, self.grid_size - 1)
        elif action == 2:
            new_col = max(new_col - 1, 0)
        elif action == 3:
            new_col = min(new_col + 1, self.grid_size - 1)
        self.agent_pos = (new_row, new_col)
        reward, done = self.calculate_reward(action)
        if self.agent_pos == self.food_pos:
            reward += 5.0
            done = True
        elif self.agent_pos == self.poison_pos:
            reward -= 5.0
            done = True
        next_state = self.get_state(position)
        return next_state, reward, done

    def calculate_reward(self, action):
        reward = -0.2
        return reward, False

def get_action_mask(agent_pos, grid_size):
    mask = torch.zeros(1, 4).to(Config.DEVICE)
    row, col = agent_pos
    if row == 0:
        mask[0, 0] = -1e9
    if row == grid_size - 1:
        mask[0, 1] = -1e9
    if col == 0:
        mask[0, 2] = -1e9
    if col == grid_size - 1:
        mask[0, 3] = -1e9
    return mask

def run_episode(env, model, epsilon=0.1, max_steps=Config.SEQUENCE_LENGTH):
    memory = env.reset()
    rewards = []
    log_probs = []
    done = False
    for step in range(max_steps):
        state_tensor = torch.tensor(np.array(memory), dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
        if random.random() < epsilon:  # Epsilon-greedy exploration
            action = random.randint(0, 3)
        else:
            output = model(state_tensor)
            probs = torch.softmax(output, dim=1)
            m = torch.distributions.Categorical(probs)
            action = m.sample().item()
            log_prob = m.log_prob(torch.tensor(action).to(Config.DEVICE))
            log_probs.append(log_prob)
        next_state, reward, done = env.step(action, step)
        rewards.append(reward)
        memory.extend(next_state)
        if done:
            break
    returns = [sum(rewards[i:]) for i in range(len(rewards))]
    return log_probs, returns, sum(rewards)

def train():
    env = GridEnv()
    model = TransformerModel().to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    for epoch in range(Config.EPOCHS):
        total_rewards = []
        total_loss = 0
        for batch in range(Config.TRAIN_STEPS // Config.BATCH_SIZE):
            batch_log_probs = []
            batch_returns = []
            for _ in range(Config.BATCH_SIZE):
                log_probs, returns, reward = run_episode(env, model, epsilon=max(0.1, 1 - epoch / Config.EPOCHS))
                batch_log_probs.extend(log_probs)
                batch_returns.extend(returns)
                total_rewards.append(reward)
            if batch_log_probs:
                batch_log_probs = torch.stack(batch_log_probs).to(Config.DEVICE)
                batch_returns = torch.tensor(batch_returns, dtype=torch.float32).to(Config.DEVICE)
                batch_returns = (batch_returns - batch_returns.mean()) / (batch_returns.std() + 1e-6)
                loss = -(batch_log_probs * batch_returns).mean()
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        print(f"Epoch {epoch}: Avg Reward: {sum(total_rewards) / len(total_rewards):.2f}, Loss: {total_loss:.4f}")

def validate_model(model, env, epoch, validation_samples=100):
    model.eval()
    validation_loss = 0.0
    successes = 0
    for _ in range(validation_samples):
        log_probs, returns, _ = run_episode(env, model, random_exploration=False)
        if env.agent_pos == env.food_pos:
            successes += 1
        if log_probs:
            returns = torch.tensor(returns, dtype=torch.float32).to(Config.DEVICE)
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-6)
            log_probs = torch.stack(log_probs).to(Config.DEVICE)
            val_loss = -(log_probs * returns).mean()
            validation_loss += val_loss.item()
    validation_loss /= validation_samples
    accuracy = successes / validation_samples * 100
    print(f"Validation {epoch}: Loss: {validation_loss:.4f} - Accuracy: {accuracy:.0f}%")
    model.train()
    return accuracy

def run_tests(num_val_tests=Config.VAL_TESTS):
    env = GridEnv()
    model = TransformerModel().to(Config.DEVICE)
    model.load_state_dict(torch.load("model.pth", map_location=Config.DEVICE))
    model.eval()
    total_successes = 0
    total_poisons = 0
    for _ in range(num_val_tests):
        memory = env.reset()
        done = False
        for idx in range(Config.MAX_AGENT_STEPS):
            state_array = np.array(memory)
            input_seq = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
            with torch.no_grad():
                output = model(input_seq)
                mask = get_action_mask(env.agent_pos, env.grid_size)
                masked_output = output + mask
                action = torch.argmax(masked_output, dim=1).item()
            next_state, _, done = env.step(action, idx % Config.SEQUENCE_LENGTH)
            memory.extend(next_state)
            if done:
                if env.agent_pos == env.food_pos:
                    total_successes += 1
                elif env.agent_pos == env.poison_pos:
                    total_poisons += 1
                break
    print(f"{total_successes} food, {total_poisons} poison, {num_val_tests} tests, {total_successes / num_val_tests * 100:.0f}% success")

def run_single():
    env = GridEnv()
    model = TransformerModel().to(Config.DEVICE)
    model.load_state_dict(torch.load("model.pth", map_location=Config.DEVICE))
    model.eval()
    memory = env.reset()
    done = False
    action_map = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}
    step_count = 0
    print(f"Start: {env.agent_pos}, Food: {env.food_pos}, Poison: {env.poison_pos}")
    while not done and step_count < Config.MAX_AGENT_STEPS:
        state_array = np.array(memory)
        input_seq = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
        with torch.no_grad():
            output = model(input_seq)
            mask = get_action_mask(env.agent_pos, env.grid_size)
            masked_output = output + mask
            action = torch.argmax(masked_output, dim=1).item()
        next_state, _, done = env.step(action, step_count % Config.SEQUENCE_LENGTH)
        memory.extend(next_state)
        print(f"{action_map[action]}: {env.agent_pos}")
        if env.agent_pos == env.food_pos:
            print("Found food!")
            done = True
        elif env.agent_pos == env.poison_pos:
            print("Stepped on poison!")
            done = True
        step_count += 1
    if not done:
        print("Maximum steps reached without stepping on anything.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', action='store_true', help="Train the model")
    parser.add_argument('-e', action='store_true', help="Evaluate")
    parser.add_argument('-s', action='store_true', help="Run single instance")
    args = parser.parse_args()
    if args.t:
        train()
    elif args.e:
        run_tests()
    elif args.s:
        run_single()
    else:
        print('Launch the file with one of the arguments: -t, -s, or -e.')

if __name__ == "__main__":
    main()

