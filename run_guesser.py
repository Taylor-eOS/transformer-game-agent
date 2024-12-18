import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from utils_guesser import print_memory

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRID_SIZE = 3
TEST_MODE = False
EVALUATION = True
EPOCHS = 2 if TEST_MODE else 8192
TRAIN_STEPS = 2 if TEST_MODE else 128
BATCH_SIZE = 2 if TEST_MODE else 64
MAX_AGENT_STEPS = 10
NUM_VAL_TESTS = 100
SEQUENCE_LENGTH = 5
STATE_VECTORS = 4
VECTOR_DIMENSION = 3
LEARNING_RATE = 0.00001
GAMMA = 0.999

class TransformerModel(nn.Module):
    def __init__(self, input_size=VECTOR_DIMENSION, seq=SEQUENCE_LENGTH * STATE_VECTORS, num_actions=4, embed_size=16, num_heads=2, num_layers=3):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq, embed_size))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, num_actions)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

class GridEnv:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.reset()

    def normalize_position(self, pos):
        return pos[0] / (self.grid_size - 1), pos[1] / (self.grid_size - 1)

    def get_state(self, position):
        state = []
        agent_x, agent_y = self.agent_pos
        food_dx, food_dy = self.food_pos[0] - agent_x, self.food_pos[1] - agent_y
        poison_dx, poison_dy = self.poison_pos[0] - agent_x, self.poison_pos[1] - agent_y
        norm_food_dx, norm_food_dy = food_dx / (self.grid_size - 1), food_dy / (self.grid_size - 1)
        norm_poison_dx, norm_poison_dy = poison_dx / (self.grid_size - 1), poison_dy / (self.grid_size - 1)
        state.append([position, 1, norm_food_dx])
        state.append([position, 1, norm_food_dy])
        state.append([position, 2, norm_poison_dx])
        state.append([position, 2, norm_poison_dy])
        return np.array(state, dtype=np.float32)

    def reset(self):
        positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        self.agent_pos = random.choice(positions)
        positions.remove(self.agent_pos)
        self.food_pos = random.choice(positions)
        positions.remove(self.food_pos)
        self.poison_pos = random.choice(positions)
        self.previous_pos = None
        initial_states = [self.get_state(0) for _ in range(SEQUENCE_LENGTH)]
        flat_initial_states = [vec for state in initial_states for vec in state]
        return deque(flat_initial_states, maxlen=SEQUENCE_LENGTH * STATE_VECTORS)

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
            reward += 4.0
            done = True
        elif self.agent_pos == self.poison_pos:
            reward -= 3.0
            done = True
        next_state = self.get_state(position)
        return next_state, reward, done

    def calculate_reward(self, action):
        reward = -0.1
        return reward, False

def get_action_mask(agent_pos, grid_size):
    mask = torch.zeros(1, 4).to(DEVICE)
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

def train():
    env = GridEnv()
    model = TransformerModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    validation_samples = 100
    for epoch in range(EPOCHS):
        total_loss = 0.0
        loss_count = 0
        batches = TRAIN_STEPS // BATCH_SIZE
        positive_count = 0
        random_exploration = False if TEST_MODE else epoch % (EPOCHS // 10) == 0
        for _ in range(batches):
            batch_log_probs = []
            batch_returns = []
            for _ in range(BATCH_SIZE):
                memory = env.reset()
                G = 0.0
                log_probs = []
                rewards = []
                done = False
                if random_exploration:
                    sequence = [random.randint(0, 3) for _ in range(SEQUENCE_LENGTH)]
                for idx in range(SEQUENCE_LENGTH):
                    state_tensor = torch.tensor(np.array(memory), dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    if random_exploration:
                        action = sequence[idx]
                    else:
                        output = model(state_tensor)
                        mask = get_action_mask(env.agent_pos, env.grid_size).to(DEVICE)
                        masked_output = output + mask
                        probs = torch.softmax(masked_output, dim=1)
                        action = torch.multinomial(probs, 1).item()
                        log_prob = torch.log(probs[0, action] + 1e-8)
                        log_probs.append(log_prob)
                    next_state, reward, done = env.step(action, idx)
                    G += reward
                    memory.extend(next_state)
                    rewards.append(reward)
                    if done:
                        break
                returns = []
                G = 0
                for reward in reversed(rewards[:len(log_probs)]):
                    G = reward + G * GAMMA
                    returns.insert(0, G)
                if not random_exploration:
                    batch_log_probs.extend(log_probs)
                    batch_returns.extend(returns)
            if not random_exploration and batch_log_probs:
                batch_log_probs = torch.stack(batch_log_probs).to(DEVICE)
                batch_returns = torch.tensor(batch_returns, dtype=torch.float32).to(DEVICE)
                std = batch_returns.std(unbiased=False)
                batch_returns = (batch_returns - batch_returns.mean()) / (std + 1e-6)
                loss = -(batch_log_probs * batch_returns).mean()
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_count += 1
                if G > 0:
                    positive_count += 1
        if loss_count > 0:
            average_loss = total_loss / loss_count
            print(f"{epoch + 1}/{EPOCHS} - Positives: {positive_count} - Training loss: {average_loss:.4f}")
        else:
            print(f"{epoch + 1}/{EPOCHS} - No updates")
        if epoch % 10 == 0:
            model.eval()
            validation_loss = 0.0
            successes = 0
            for _ in range(validation_samples):
                memory = env.reset()
                G = 0.0
                log_probs = []
                rewards = []
                done = False
                for idx in range(SEQUENCE_LENGTH):
                    state_tensor = torch.tensor(np.array(memory), dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    output = model(state_tensor)
                    mask = get_action_mask(env.agent_pos, env.grid_size).to(DEVICE)
                    masked_output = output + mask
                    probs = torch.softmax(masked_output, dim=1)
                    action = torch.argmax(probs, dim=1).item()
                    log_prob = torch.log(probs[0, action] + 1e-8)
                    log_probs.append(log_prob)
                    next_state, reward, done = env.step(action, idx)
                    G += reward
                    memory.extend(next_state)
                    rewards.append(reward)
                    if done:
                        if env.agent_pos == env.food_pos:
                            successes += 1
                        break
                returns = []
                G = 0
                for reward in reversed(rewards[:len(log_probs)]):
                    G = reward + G * GAMMA
                    returns.insert(0, G)
                returns = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
                std = returns.std(unbiased=False)
                returns = (returns - returns.mean()) / (std + 1e-6)
                log_probs = torch.stack(log_probs).to(DEVICE)
                val_loss = -(log_probs * returns).mean()
                validation_loss += val_loss.item()
            validation_loss /= validation_samples
            accuracy = successes / validation_samples * 100
            print(f"Validation loss: {validation_loss:.4f} - Validation accuracy: {accuracy:.0f}%")
            model.train()
    torch.save(model.state_dict(), "model.pth")

def run_tests(num_val_tests=NUM_VAL_TESTS):
    env = GridEnv()
    model = TransformerModel().to(DEVICE)
    model.load_state_dict(torch.load("model.pth", map_location=torch.device(DEVICE), weights_only=True))
    model.eval()
    total_successes = 0
    total_poisons = 0
    for _ in range(num_val_tests):
        memory = env.reset()
        done = False
        for idx in range(MAX_AGENT_STEPS):
            state_array = np.array(memory)
            input_seq = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(input_seq)
                probs = torch.softmax(output, dim=1)
                mask = get_action_mask(env.agent_pos, env.grid_size).to(DEVICE)
                masked_probs = probs + mask
                action = torch.argmax(masked_probs, dim=1).item()
            next_state, reward, done = env.step(action, idx % SEQUENCE_LENGTH)
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
    model = TransformerModel().to(DEVICE)
    model.load_state_dict(torch.load("model.pth", map_location=torch.device(DEVICE), weights_only=True))
    model.eval()
    memory = env.reset()
    done = False
    action_map = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}
    step_count = 0
    print(f"Start: {env.agent_pos}, Food: {env.food_pos}, Poison: {env.poison_pos}")
    while not done and step_count < MAX_AGENT_STEPS:
        state_array = np.array(memory)
        input_seq = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_seq)
            probs = torch.softmax(output, dim=1)
            mask = get_action_mask(env.agent_pos, env.grid_size).to(DEVICE)
            masked_probs = probs + mask
            action = torch.argmax(masked_probs, dim=1).item()
        next_state, reward, done = env.step(action, step_count % SEQUENCE_LENGTH)
        memory.extend(next_state)
        if TEST_MODE: print_memory(memory, GRID_SIZE)
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
    parser.add_argument('-s', action='store_true', help="Run single instance")
    parser.add_argument('-e', action='store_true', help="Evaluate")
    args = parser.parse_args()
    if args.t:
        train()
    elif args.s:
        run_single()
    elif args.e:
        run_tests()
    else:
        print('Launch the file with one of the arguments: -t, -s, or -e.')

if __name__ == "__main__":
    main()

