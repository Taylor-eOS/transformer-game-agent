import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from collections import namedtuple

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRID_SIZE = 3
TEST_MODE = False
EVALUATION = True
EPOCHS = 1 if TEST_MODE else 200
TRAIN_STEPS = 1 if TEST_MODE else 100
MAX_AGENT_STEPS = 10
NUM_VAL_TESTS = 100
SEQUENCE_LENGTH = 5
EMBED_SIZE = 32
LAYERS = 2
INPUT_SIZE = 11

class TransformerModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, sequence_length=SEQUENCE_LENGTH, num_actions=4, embed_size=EMBED_SIZE, num_heads=2, num_layers=LAYERS):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, sequence_length, embed_size))
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

    def get_state(self, position):
        state = np.zeros(INPUT_SIZE, dtype=np.float32)
        state[0] = position
        state[1] = self.food_pos[0]
        state[2] = self.food_pos[1]
        state[3] = self.poison_pos[0]
        state[4] = self.poison_pos[1]
        state[5] = self.agent_pos[0]
        state[6] = self.agent_pos[1]
        state[7] = self.food_pos[0] - self.agent_pos[0]
        state[8] = self.food_pos[1] - self.agent_pos[1]
        state[9] = self.poison_pos[0] - self.agent_pos[0]
        state[10] = self.poison_pos[1] - self.agent_pos[1]
        return state

    def reset(self):
        positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        self.agent_pos = random.choice(positions)
        positions.remove(self.agent_pos)
        self.food_pos = random.choice(positions)
        positions.remove(self.food_pos)
        self.poison_pos = random.choice(positions)
        self.previous_pos = None
        return self.get_state(0)

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
        return self.get_state(position), reward, done

    def calculate_reward(self, action):
        reward = -0.1
        #reward += self.step_back_penalty()
        return reward, False

    def step_back_penalty(self):
        if self.previous_pos and self.agent_pos == self.previous_pos:
            #if TEST_MODE: print('Back step')
            return -0.2
        return 0.0

def get_action_mask(agent_pos, grid_size):
    mask = torch.zeros(1,4)
    row, col = agent_pos
    if row == 0:
        mask[0,0] = -1e9
    if row == grid_size -1:
        mask[0,1] = -1e9
    if col == 0:
        mask[0,2] = -1e9
    if col == grid_size -1:
        mask[0,3] = -1e9
    return mask

def generate_sequence(sequence_length=SEQUENCE_LENGTH):
    sequence = [random.randint(0,3) for _ in range(sequence_length)]
    if TEST_MODE: print(sequence)
    return sequence

def train():
    env = GridEnv()
    model = TransformerModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.000001)
    for epoch in range(EPOCHS):
        total_loss = 0.0
        loss_count = 0
        for _ in range(TRAIN_STEPS):
            state = env.reset()
            memory = deque([env.get_state(i) for i in range(SEQUENCE_LENGTH)], maxlen=SEQUENCE_LENGTH)
            sequence = generate_sequence(SEQUENCE_LENGTH)
            G = 0.0
            log_probs = []
            done = False
            for idx, action in enumerate(sequence):
                state_tensor = torch.tensor(np.array(memory), dtype=torch.float32).unsqueeze(0).to(DEVICE)
                output = model(state_tensor)
                probs = torch.softmax(output, dim=1)
                mask = get_action_mask(env.agent_pos, env.grid_size).to(DEVICE)
                masked_probs = probs + mask
                action_prob = masked_probs[0, action]
                log_prob = torch.log(action_prob)
                log_probs.append(log_prob)
                next_state, reward, done = env.step(action, idx)
                G += reward
                memory.append(next_state)
                if done:
                    break
            for log_prob in log_probs:
                total_loss += -log_prob * G
                loss_count += 1
        if loss_count > 0:
            total_loss = total_loss / loss_count
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"{epoch +1}/{EPOCHS}")
        else:
            print(f"{epoch +1}/{EPOCHS} - No updates")
        if EVALUATION and epoch % (EPOCHS // 10) == 0 and not epoch > EPOCHS - 9:
            torch.save(model.state_dict(), "model.pth")
            run_tests(50)
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
        state = env.reset()
        food_pos = env.food_pos
        poison_pos = env.poison_pos
        agent_pos = env.agent_pos
        memory = deque([env.get_state(i) for i in range(SEQUENCE_LENGTH)], maxlen=SEQUENCE_LENGTH)
        done = False
        for idx in range(MAX_AGENT_STEPS):
            state_array = np.array(memory)
            input_seq = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(input_seq)
                probs = torch.softmax(output, dim=1)
                mask = get_action_mask(agent_pos, env.grid_size).to(DEVICE)
                masked_probs = probs + mask
                action = torch.argmax(masked_probs, dim=1).item()
            next_state, reward, done = env.step(action, idx % SEQUENCE_LENGTH)
            agent_pos = env.agent_pos
            memory.append(next_state)
            if done:
                if agent_pos == food_pos:
                    total_successes += 1
                elif agent_pos == poison_pos:
                    total_poisons += 1
                break
    print(f"{total_successes} successes, {total_poisons} fails, {num_val_tests} tests, {total_successes / num_val_tests * 100:.0f}%")

def run_single():
    env = GridEnv()
    model = TransformerModel().to(DEVICE)
    model.load_state_dict(torch.load("model.pth", map_location=torch.device(DEVICE), weights_only=True))
    model.eval()
    memory = deque([env.reset() for _ in range(SEQUENCE_LENGTH)], maxlen=SEQUENCE_LENGTH)
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
        agent_pos = env.agent_pos
        memory.append(next_state)
        print(f"{action_map[action]}: {agent_pos}")
        if agent_pos == env.food_pos:
            print("Found food!")
            done = True
        elif agent_pos == env.poison_pos:
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

