import random

def print_memory(memory, grid_size):
    def unnormalize(value):
        return int(value * (grid_size - 1))
    print("Memory Contents:")
    memory_list = list(memory)
    for i in range(0, len(memory_list), 3):
        timestep = memory_list[i:i+3]
        print(f"Timestep {i // 3}:")
        agent_vec = timestep[0]
        food_vec = timestep[1]
        poison_vec = timestep[2]
        print(f"  Agent Vector   : [{int(agent_vec[0])}, {int(agent_vec[1])}, {unnormalize(agent_vec[2])}, {unnormalize(agent_vec[3])}]")
        print(f"  Food Vector    : [{int(food_vec[0])}, {int(food_vec[1])}, {unnormalize(food_vec[2])}, {unnormalize(food_vec[3])}]")
        print(f"  Poison Vector  : [{int(poison_vec[0])}, {int(poison_vec[1])}, {unnormalize(poison_vec[2])}, {unnormalize(poison_vec[3])}]")

def generate_sequence(sequence_length):
    sequence = [random.randint(0,3) for _ in range(sequence_length)]
    #print(sequence)
    return sequence

def generate_sequence_rest(sequence_length, GRID_SIZE):
    directions = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}
    sequence = []
    current_position = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
    last_move = 1
    for _ in range(sequence_length):
        possible_moves = []
        for move, direction in directions.items():
            if move == (last_move + 2) % 4:
                continue
            next_position = None
            if direction == "Up" and current_position[0] > 0:
                next_position = (current_position[0] - 1, current_position[1])
            elif direction == "Down" and current_position[0] < GRID_SIZE - 1:
                next_position = (current_position[0] + 1, current_position[1])
            elif direction == "Left" and current_position[1] > 0:
                next_position = (current_position[0], current_position[1] - 1)
            elif direction == "Right" and current_position[1] < GRID_SIZE - 1:
                next_position = (current_position[0], current_position[1] + 1)
            if next_position:
                possible_moves.append((move, next_position))
        if not possible_moves:
            break
        chosen_move, next_position = random.choice(possible_moves)
        sequence.append(chosen_move)
        current_position = next_position
        last_move = chosen_move
    print(sequence)
    return sequence

