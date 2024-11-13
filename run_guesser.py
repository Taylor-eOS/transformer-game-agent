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
EPOCHS = 1 if TEST_MODE else 64
TRAIN_STEPS = 1 if TEST_MODE else 512
BATCH_SIZE = 32
MAX_AGENT_STEPS = 10
NUM_VAL_TESTS = 100
EMBED_SIZE = 32
SEQUENCE_LENGTH = 5
INPUT_SIZE = 4
LEARNING_RATE = 0.0001

class TransformerModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, seq=SEQUENCE_LENGTH * 3, num_actions=4, embed_size=EMBED_SIZE, num_heads=2, num_layers=2):
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
        norm_x, norm_y = self.normalize_position(self.agent_pos)
        state.append([position, 0, norm_x, norm_y])
        norm_x, norm_y = self.normalize_position(self.food_pos)
        state.append([position, 1, norm_x, norm_y])
        norm_x, norm_y = self.normalize_position(self.poison_pos)
        state.append([position, 2, norm_x, norm_y])
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
        return deque(flat_initial_states, maxlen=SEQUENCE_LENGTH * 3)

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
            reward += 3.0
            done = True
        elif self.agent_pos == self.poison_pos:
            reward -= 2.0
            done = True
        next_state = self.get_state(position)
        #print(reward)
        return next_state, reward, done

    def calculate_reward(self, action):
        reward = -0.1
        return reward, False

def get_action_mask(agent_pos, grid_size):
    mask = torch.zeros(1, 4).to(DEVICE)  # Ensure mask is on DEVICE
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

def generate_sequence(sequence_length=SEQUENCE_LENGTH):
    sequence = [random.randint(0,3) for _ in range(sequence_length)]
    return sequence

def train():
    env = GridEnv()
    model = TransformerModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        total_loss = 0.0
        loss_count = 0
        batches = TRAIN_STEPS // BATCH_SIZE
        positive_count = 0
        for _ in range(batches):
            batch_log_probs = []
            batch_returns = []
            for _ in range(BATCH_SIZE):
                memory = env.reset()
                sequence = generate_sequence(SEQUENCE_LENGTH)
                G = 0.0
                log_probs = []
                done = False
                for idx, action in enumerate(sequence):
                    state_tensor = torch.tensor(np.array(memory), dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    output = model(state_tensor)
                    mask = get_action_mask(env.agent_pos, env.grid_size).to(DEVICE)
                    masked_output = output + mask
                    probs = torch.softmax(masked_output, dim=1)
                    action_prob = probs[0, action]
                    log_prob = torch.log(action_prob + 1e-8)
                    log_probs.append(log_prob)
                    next_state, reward, done = env.step(action, idx)
                    G += reward
                    memory.extend(next_state)
                    if done:
                        break
                batch_log_probs.extend(log_probs)
                batch_returns.append(G)
            loss = -torch.stack(batch_log_probs).mean() * torch.tensor(batch_returns).mean()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_count += 1
            if G > 0:
                positive_count += 1
        if loss_count > 0:
            average_loss = total_loss / loss_count
            print(f"{epoch +1}/{EPOCHS} - {positive_count} - {average_loss:.2f}")
        else:
            print(f"{epoch +1}/{EPOCHS} - No updates")
        if EVALUATION and not TEST_MODE and epoch % (EPOCHS // 10) == 0 and not epoch > EPOCHS - 9:
            torch.save(model.state_dict(), "model.pth")
            run_tests(100)
    torch.save(model.state_dict(), "model.pth")
    if not TEST_MODE:
        run_tests(100)

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
    parser.add_argument('--t', action='store_true')
    parser.add_argument('--s', action='store_true')
    args = parser.parse_args()
    if args.t:
        train()
    elif args.s:
        run_single()
    else:
        run_tests()

if __name__ == "__main__":
    main()

