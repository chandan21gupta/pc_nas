def parse_actions_index(actions_index, steps):
    actions_index_ = actions_index.detach().numpy()
    blocks = []
    operations = []
    i = 0
    while(i < len(actions_index_)):
        curr_block = []
        for j in range(steps):
            curr_block.append(actions_index_[i+j])
        operations.append(actions_index_[i+j])
        blocks.append(curr_block)
        i = i+j+1
    return blocks, operations

    

