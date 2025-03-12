"""
Preprocess dataset for Overcooked environment.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
print(sys.path)
import numpy as np
import os
from datasets import Dataset
import argparse
from ragen.env.overcooked.prompts import *

from ragen.env.overcooked.env import OverCookedEnv
from transformers import AutoTokenizer
LLM_system_prompt = "You are a friendly chat assistant who is correct and brief at all times."

Rules = f'''Players must coordinate to make onion soups with 3 onions each. Once a soup is cooked it needs to be placed on a plate and delivered. Players can only carry one item at a time. A soup can only be loaded onto plate by a player if they are holding a plate. The goal is to maximize the number of deliveries.'''


Base_prompt={
    'deepseek': """I am {player_name}. I am playing the game Overcooked with my partner {other_player_name}. {envDescription} """ + f'''
Overcooked has the following rules: {Rules}. We have agreed to be efficient and prepare for the next soup while the current soup is cooking. I'll provide my action history, current state, teammate's status, and my possible actions. Help me select the best action from the list. Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format''',
    'qwen-instruct': """I am {player_name}. I am playing the game Overcooked with my partner {other_player_name}. {envDescription} """ + f'''
Overcooked has the following rules: {Rules}. We have agreed to be efficient and prepare for the next soup while the current soup is cooking. I'll provide my action history, current state, teammate's status, and my possible actions. Help me select the best action from the list. Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format'''
}

def Hashing(seed):
    return seed * seed % len(OverCookedEnv.LAYOUTS)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate trajectories using specified environment and policy.")
    parser.add_argument("--env", type=str, default="overcooked", help="Environment name (default: 'sokoban').")
    parser.add_argument("--seed", type=int, default=10000, help="Seed for random number generation (default: 10000).")
    parser.add_argument("--output", type=str, default="data/overcooked", help="Output file to save the trajectories (default: 'data/sokoban').")
    parser.add_argument("--train_size", type=int, default=300, help="Number of trajectories to generate (default: 3000).")
    parser.add_argument("--test_size", type=int, default=10, help="Number of trajectories to generate (default: 100).")
    parser.add_argument("--bfs_max_nodes", type=int, default=1000, help="Maximum number of nodes to use for BFS (default: 100000).") # not using this now. This will usually give the best traj. To compare with SFT, we will try this later.
    parser.add_argument("--prefix", type=str, default='qwen-instruct', choices=['qwen-instruct', 'deepseek'])
    
    args = parser.parse_args()
    if args.prefix == 'deepseek':
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    else:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    assert args.env == "overcooked", "Unsupported environment: {args.env}"
    
    seeds = range(args.seed, args.seed + args.train_size + args.test_size)
    instructions = []
    for seed in seeds:
        layout_name = OverCookedEnv.LAYOUTS[Hashing(seed)]
        env = OverCookedEnv(layout_name)
        Message_Alice = [
                    {"role": "system", "content": LLM_system_prompt},
                    {"role": "user", "content": Base_prompt[args.prefix].format(player_name="Alice", other_player_name="Bob", envDescription=EnvDescriptions[layout_name])},
                    {"role": "user", "content": env.am[0].llm_agent._state_to_description(env.am[0].prepare_next_move(env.state))},
                ]
        Message_Bob = [
                    {"role": "system", "content": LLM_system_prompt},
                    {"role": "user", "content": Base_prompt[args.prefix].format(player_name="Bob", other_player_name="Alice", envDescription=EnvDescriptions[layout_name])},
                    {"role": "user", "content": env.am[1].llm_agent._state_to_description(env.am[1].prepare_next_move(env.state))},
                ]
        if args.prefix == 'deepseek':
            instructions.append([
                tokenizer.apply_chat_template(Message_Alice, add_generation_prompt=True, tokenize=False), 
                tokenizer.apply_chat_template(Message_Bob, add_generation_prompt=True, tokenize=False)
            ])
        else:
            instructions.append([
                tokenizer.apply_chat_template(Message_Alice, add_generation_prompt=False, tokenize=False)+ '<|im_start|>assistant\n<think>',
                tokenizer.apply_chat_template(Message_Bob, add_generation_prompt=False, tokenize=False) + '<|im_start|>assistant\n<think>'
            ])
    
    
    def _create_instance(idx, instruction):
        return {
            "data_source": "overcooked",
            "prompt": {"content": instruction},
            "ability": "bfs",
            "reward_model": {"style": "rule", "ground_truth": {"target": 0, "numbers": [0, 0]}},
            "extra_info": {"split": "train", "index": idx}
        }
    train_dataset = Dataset.from_list([_create_instance(Hashing(args.seed + i), instructions[i]) for i in range(args.train_size)])
    test_dataset = Dataset.from_list([_create_instance(Hashing(args.seed + i), instructions[i]) for i in range(args.train_size, args.train_size + args.test_size)])


    def make_map_fn(split):
        def process_fn(example, idx):
            return example
        return process_fn

    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    train_dataset.to_parquet(os.path.join(args.output, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.output, 'test.parquet'))
    # print one instance of train dataset
    print(train_dataset[0])

if __name__ == "__main__":
    main()