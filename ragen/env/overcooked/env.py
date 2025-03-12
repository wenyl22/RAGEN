import sys
import os
import re
sys.path.append(os.path.join(os.path.dirname(__file__)))
from typing_extensions import override
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.actions import Action
from ragen.env.overcooked.action_manager import LLMActionManager
import numpy as np 
import random
from ragen.env.base import BaseDiscreteActionEnv
from fuzzywuzzy import process

def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    
class OverCookedEnv(BaseDiscreteActionEnv):
    LAYOUTS = [
        'cramped_room',
        'no_counter_door',
        'soup_passing',
        'soup_passing_door',
        'asymmetric_advantages',
        'forced_coordination',
        'coordination_ring',
        'counter_circuit_o_1order',
        'bottleneck',
        'large_room',
        'centre_objects',
        'centre_pots',
    ]
    def __init__(self, layout_name):
        BaseDiscreteActionEnv.__init__(self)
        self.num_env_steps = 0
        self.max_steps = 400
        self.INVALID_ACTION = Action.STAY
        self.layout_name = layout_name
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.state = self.mdp.get_standard_start_state()
        self.am = [
            LLMActionManager(self.mdp, 'player_0', self.layout_name),
            LLMActionManager(self.mdp, 'player_1', self.layout_name)
        ]
        self.available_actions_list = [[], []]
    
    def step(self, joint_action):
        """
        - Step the environment with the given action.
        - Check if the action is effective (whether player moves in the env).
        """
        self.num_env_steps += 1
        prev_state = self.state
        (self.state, 
        sparse_reward, 
        shaped_reward, 
        sparse_reward_by_agent, 
        shaped_reward_by_agent) = self.mdp.get_state_transition(
            prev_state, 
            joint_action,
        )
        obs = []
        for i in range(2):
            state_for_llm = self.am[i].prepare_next_move(self.state)
            obs.append(self.am[i].llm_agent._state_to_description(state_for_llm, ""))
        self.num_env_steps += 1
        done = self.finished()
        return obs, shaped_reward, done, {"action_is_effective": True}
    def finished(self):
        return self.num_env_steps >= self.max_steps
    def success(self):
        return self.num_env_steps >= self.max_steps
    def render(self, mode='tiny_rgb_array'):
        raise ValueError("OverCookedEnv does not support rendering yet.")    
    def copy(self):
        raise ValueError("OverCookedEnv does not support copy, just make a new one.")
    def reset(self):
        raise ValueError("OverCookedEnv does not support reset, just make a new one.")
    def extract_action(self, text, player_id):
        print(f"Extracting action from {text}")
        if "</think>" not in text:
            return self.INVALID_ACTION, False        
        text = text.split("</think>")[1]
        match = re.search(r"<answer>(.*?)</answer>", text)
        if not match:
            return self.INVALID_ACTION, False
        selected_match = match.group(1).strip()
        state_for_llm = self.am[player_id].prepare_next_move(self.state)
        self.available_actions_list[player_id] = self.am[player_id].llm_agent._get_available_actions(state_for_llm, '')
        for action in self.available_actions_list[player_id]:
            if selected_match.lower() in action.lower():
                return action, True 
        selected_move, score = process.extractOne(selected_match, self.available_actions_list[player_id])
        selected_action, message = self.am[player_id].make_next_move(selected_move, selected_move)
        return selected_action, True

    def formulate_output(self, env_feedback, done, tokenizer):
        if not done:
            if "qwen" in tokenizer.name_or_path.lower():
                output = "\n <|im_start|>user\n" + env_feedback + "<|im_end|>\n"
                output += "<|im_start|>assistant\n<think>"
            else:
                assert "deepseek" in tokenizer.name_or_path.lower(), "Only Qwen and DeepSeek models are hard coded"
                output = tokenizer.apply_chat_template([{"role": "user", "content": env_feedback}], tokenize=False, add_generation_prompt=True)

        return output
            
    def execute_predictions(envs, predictions, prediction_ids, tokenizer, effective_player_ids):
        # parse the responses [:, 2, :] and get the joint action in game space
        cur_actions = [[Action.STAY, Action.STAY] for _ in range(len(envs))]
        action_is_valid = [[False, False] for _ in range(len(envs))]
        for i, env in enumerate(envs):
            for _ in range(2):
                cur_actions[i][_], 
                action_is_valid[i][_] = env.extract_action(predictions[_][i], _)
        
        next_obs, dones = [], []
        
        for i, (env, action, av) in enumerate(zip(envs, cur_actions, action_is_valid)):
            keep_id = effective_player_ids[i]
            obs = [tokenizer.eos_token if tokenizer.eos_token not in predictions[_][i] else "" for _ in range(2)]
            if env.finished():
                obs = [obs[_] + tokenizer.pad_token for _ in range(2)]
                done = True
            else:
#                thinking_reward = [0, 0]
                # step in environment
                observation, env_reward, done, extra_info = env.step(action)
                for _ in range(2):
                    observation[_] = env.formulate_output(observation[_], done, tokenizer)
                    obs[_] += observation[_]
                env._update_tracking_variables(
                    response=predictions[keep_id][i],
                    action=action[keep_id],
                    action_is_valid=av[keep_id],
                    action_is_effective=extra_info.get("action_is_effective", False), 
                    reward=env_reward,
                )
                
            next_obs.append(obs)
            dones.append(done)
        return next_obs, dones