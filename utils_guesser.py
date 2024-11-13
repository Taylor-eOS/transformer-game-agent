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

