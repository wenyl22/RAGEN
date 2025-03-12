import logging
from overcooked_ai_py.mdp.actions import LLMActionSet
from ragen.env.overcooked.prompts import *
logging.basicConfig(filename='debug.log', level=logging.DEBUG)
logging.debug('Initiated Logger...')
STAY_COMPLETION="Action: wait."

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class LLMAgent:
    def __init__(self, player_id, layout_name):
        self.player_id = player_id
        self.layout_name = layout_name
        self.player_names = ['Alice', 'Bob']
        # Controls 
        self.DEBUG = False      
        # Enable kitchen counters only for GPT-4, other models cannot handle the complexity
        self.enable_kitchen_counters = True   
        self.single_agent_ablation = True  

        self.action_set = LLMActionSet[self.layout_name]
        self.player_actions = []
        self.action_history = []

        # Set other player ID 
        if int(self.player_id) == 0:
            self.other_player_id = 1
        else:
            self.other_player_id = 0

        self.all_actions = []
        for key, value in self.action_set.items():
            if isinstance(value, list):
                self.all_actions.extend(value)
    
    def _get_available_actions(self, state_for_llm, message):
        # Available Action Constraints
        available_actions = []
        # Check what player is holding 
        if state_for_llm[self.player_id]['held_object'] == "nothing":
            for idx, d in enumerate(state_for_llm['distances']['onion_dispenser']):
                if d[0] not in ['infinite']:
                    available_actions.append(self.action_set['onion_dispenser'][idx])

            for idx, d in enumerate(state_for_llm['distances']['plate_dispenser']):
                if d[0] not in ['infinite']:
                    available_actions.append(self.action_set['plate'][idx])
            
            if self.enable_kitchen_counters:
            
                for idx, d in enumerate(state_for_llm['distances']['kitchen_counter']):
                    if d[0] not in ['infinite'] and state_for_llm['kitchen_counter_objects'][idx] == 'onion':
                        available_actions.append(self.action_set['kitchen_counter_pick_onion'][idx])
                    if d[0] not in ['infinite'] and state_for_llm['kitchen_counter_objects'][idx] == 'plate':
                        available_actions.append(self.action_set['kitchen_counter_pick_plate'][idx])
                    if d[0] not in ['infinite'] and state_for_llm['kitchen_counter_objects'][idx] == 'soup in plate':
                        available_actions.append(self.action_set['kitchen_counter_pick_soup'][idx])


            if 'storage_counter_pick_onion' in self.action_set:
                for idx, d in enumerate(state_for_llm['distances']['storage_counter']):
                    if d[0] not in ['infinite']:
                        if state_for_llm['storage_counter_objects'][idx] == 'onion':
                            available_actions.append(self.action_set['storage_counter_pick_onion'][idx])
                        elif state_for_llm['storage_counter_objects'][idx] == 'plate':
                            available_actions.append(self.action_set['storage_counter_pick_plate'][idx])
                        elif state_for_llm['storage_counter_objects'][idx] == 'soup in plate':
                            available_actions.append(self.action_set['storage_counter_pick_soup'][idx])
            
            for idx, d in enumerate(state_for_llm['distances']['gate']):
                if d[0] not in ['infinite']:
                    if state_for_llm['gate_status'][idx] == 'closed':
                        available_actions.append(self.action_set['gate'][idx])
        elif state_for_llm[self.player_id]['held_object'] == 'onion':
            for idx, d in enumerate(state_for_llm['distances']['cooker']):
                if d[0] not in ['infinite']:
                    available_actions.append(self.action_set['cooker'][idx])

            if self.enable_kitchen_counters:
                if len(self.empty_kitchen_counters)>0:
                    
                    kidx = self.empty_kitchen_counter_distances.index(min(self.empty_kitchen_counter_distances))
                    
                    k_action = self.action_set['kitchen_counter_place_onion'][kidx]
                    
                    available_actions.append(k_action)

            if 'storage_counter_place_onion' in self.action_set:
                for idx, d in enumerate(state_for_llm['distances']['storage_counter']):
                    if d[0] not in ['infinite']:
                        if state_for_llm['storage_counter_objects'][idx] == 'empty':
                            available_actions.append(self.action_set['storage_counter_place_onion'][idx])

        elif state_for_llm[self.player_id]['held_object'] == 'plate':
            for idx, d in enumerate(state_for_llm['distances']['cooker']):
                if d[0] not in ['infinite']:
                    available_actions.append(self.action_set['cooked_soup'][idx])
            
            if self.enable_kitchen_counters:
                if len(self.empty_kitchen_counters)>0:
                    kidx = self.empty_kitchen_counter_distances.index(min(self.empty_kitchen_counter_distances))
                    
                    k_action = self.action_set['kitchen_counter_place_plate'][kidx]
                    
                    available_actions.append(k_action)

            if 'storage_counter_place_plate' in self.action_set:
                for idx, d in enumerate(state_for_llm['distances']['storage_counter']):
                    if d[0] not in ['infinite']:
                        if state_for_llm['storage_counter_objects'][idx] == 'empty':
                            available_actions.append(self.action_set['storage_counter_place_plate'][idx])

        elif state_for_llm[self.player_id]['held_object'] == 'soup in plate':
            for idx, d in enumerate(state_for_llm['distances']['delivery_zone']):
                if d[0] not in ['infinite']:
                    available_actions.append(self.action_set['delivery_area'][idx])
            if self.enable_kitchen_counters:
                if len(self.empty_kitchen_counters)>0:
                    kidx = self.empty_kitchen_counter_distances.index(min(self.empty_kitchen_counter_distances))
                    
                    k_action = self.action_set['kitchen_counter_place_plate'][kidx]
                    
                    available_actions.append(k_action)

            if 'storage_counter_place_soup' in self.action_set:
                for idx, d in enumerate(state_for_llm['distances']['storage_counter']):
                    if d[0] not in ['infinite']:
                        if state_for_llm['storage_counter_objects'][idx] == 'empty':
                            available_actions.append(self.action_set['storage_counter_place_soup'][idx])
        return available_actions + self.action_set['wait'] + self.action_set['collision_avoidance']

    def _correct_dish_to_plate(self, state_for_llm):
        if state_for_llm[self.player_id ]['held_object'] == 'dish':
            state_for_llm[self.player_id ]['held_object'] = 'plate'
        
        if state_for_llm[self.other_player_id]['held_object'] == 'dish':
            state_for_llm[self.player_id ]['held_object'] = 'plate'
        return state_for_llm
    
    def _add_history(self):
        description = f'''action history: {', '.join(self.action_history[-5:])}.\n'''
        return description

    def _add_held_object_info(self, state_for_llm):

        description = f'''<Inventory>: I am holding {state_for_llm[self.player_id ]['held_object']}. {self.player_names[self.other_player_id]} is holding {state_for_llm[self.other_player_id ]['held_object']}. '''
        if self.single_agent_ablation:
            description = f'''<Inventory>: I am holding {state_for_llm[self.player_id ]['held_object']}. '''
        return description
    
    def _add_kitchen_facility_info_single_agent_ablation(self, state_for_llm):
        self.empty_kitchen_counters = []
        self.empty_kitchen_counter_distances = []
        description = f"<My location information:> "
        for obj_type in ['onion_dispenser', 'plate_dispenser', 'delivery_zone', 'cooker', 'storage_counter', 'gate']:
            for idx, d in enumerate(state_for_llm['distances'][obj_type]):
                if d[0] == 'infinite':
                    description += f"{obj_type[0]}{idx} is inaccessible. "
                elif 'blocked' in d[0]:
                    description += f"{obj_type[0]}{idx} is {d[0]}"
                else:
                    description += f"{obj_type[0]}{idx} is {d[0]} units away. "
            
        description += f"\n<Environment Details>: "
        for obj_type in ['cooker', 'storage_counter', 'kitchen_counter', 'gate']:
            for idx, d in enumerate(state_for_llm['distances'][obj_type]):
                if obj_type == 'cooker':
                        description += f"c{idx} contains {state_for_llm['num_onions_in_pot'][idx]} out of 3 onions. "
                if self.enable_kitchen_counters:
                    if obj_type == 'kitchen_counter':
                        if state_for_llm['kitchen_counter_objects'][idx] != 'empty':
                            if d[0] == 'infinite':
                                description += f'k{idx} is inaccessible. '
                            elif 'blocked' in d[0]:
                                description += f"k{idx} is {d[0]} " 
                            else:
                                description += f"k{idx} is {d[0]} units away. "
                                description += f"k{idx} contains {state_for_llm['kitchen_counter_objects'][idx]}. " 
                            self.empty_kitchen_counter_distances.append(float('inf'))
                        else:
                            if d[0] in ['infinite'] or 'blocked' in d[0]:
                                self.empty_kitchen_counter_distances.append(float('inf'))
                            else:
                                self.empty_kitchen_counter_distances.append(int(d[0]))
                                self.empty_kitchen_counters.append(f'k{idx}')
                
                if obj_type == 'gate':
                    if d[0] not in ['infinite']:
                        description += f"g{idx} is {state_for_llm['gate_status'][idx]}. "
                        if state_for_llm['gate_status'][idx] == 'open':
                            description += f"g{idx} will stay open for {10 - state_for_llm['gate_open_time'][idx]} timesteps. "

                if self.layout_name in ['forced_coordination', 'counter_circuit_o_1order', 'soup_passing']:   
                    if obj_type == 'storage_counter':
                        if state_for_llm['storage_counter_objects'][idx] == 'empty':
                            description += f"s{idx} is empty. "
                        else:
                            description += f"s{idx} contains {state_for_llm['storage_counter_objects'][idx]}. "
        if self.enable_kitchen_counters:
            if len(self.empty_kitchen_counter_distances) > 0:
                closest_kitchen_counter = self.empty_kitchen_counter_distances.index(min(self.empty_kitchen_counter_distances))
                distance_to_closest_kitchen_counter = min(self.empty_kitchen_counter_distances)
                if distance_to_closest_kitchen_counter != float('inf'):
                    description += f'Closest empty kitchen counter k{closest_kitchen_counter} is {distance_to_closest_kitchen_counter} units away. '

        return description


    def _add_kitchen_facility_info(self, state_for_llm):
        self.empty_kitchen_counters = []
        self.empty_kitchen_counter_distances = []
        description = f"<My location information:> "
        for obj_type in ['onion_dispenser', 'plate_dispenser', 'delivery_zone', 'cooker', 'storage_counter', 'gate']:
            for idx, d in enumerate(state_for_llm['distances'][obj_type]):
                if d[0] == 'infinite':
                    description += f"{obj_type[0]}{idx} is inaccessible. "
                elif 'blocked' in d[0]:
                    description += f"{obj_type[0]}{idx} is {d[0]} by {self.player_names[self.other_player_id]}. "
                else:
                    description += f"{obj_type[0]}{idx} is {d[0]} units away. " 
                       
        if not self.single_agent_ablation:
            description += f"\n<{self.player_names[self.other_player_id]}'s location information>: "
            for obj_type in ['onion_dispenser', 'plate_dispenser', 'delivery_zone', 'cooker', 'storage_counter', 'gate']:
                for idx, d in enumerate(state_for_llm['distances'][obj_type]):
                    if d[1] == 'infinite':
                        description += f"{obj_type[0]}{idx} is inaccessible. "
                    elif 'blocked' in d[1]:
                        description += f"{obj_type[0]}{idx} is {d[0]} by {self.player_names[self.player_id]}. "  
                    else:
                        description += f"{obj_type[0]}{idx} is {d[1]} units away. "
                    
            
        description += f"\n<Environment Details>: "
        for obj_type in ['cooker', 'storage_counter', 'kitchen_counter', 'gate']:
            for idx, d in enumerate(state_for_llm['distances'][obj_type]):
                if obj_type == 'cooker':
                        description += f"c{idx} contains {state_for_llm['num_onions_in_pot'][idx]} out of 3 onions. "
                if self.enable_kitchen_counters:
                    if obj_type == 'kitchen_counter':
                        if state_for_llm['kitchen_counter_objects'][idx] != 'empty':
                            if d[0] == 'infinite':
                                description += f'k{idx} is inaccessible. '
                            elif 'blocked' in d[0]:
                                description += f"k{idx} is {d[0]} by {self.player_names[int(self.other_player_id)]}. " 
                            else:
                                description += f"k{idx} is {d[0]} units away. "
                                description += f"k{idx} contains {state_for_llm['kitchen_counter_objects'][idx]}. " 
                            self.empty_kitchen_counter_distances.append(float('inf'))
                        else:
                            if d[0] in ['infinite'] or 'blocked' in d[0]:
                                self.empty_kitchen_counter_distances.append(float('inf'))
                            else:
                                self.empty_kitchen_counter_distances.append(int(d[0]))
                                self.empty_kitchen_counters.append(f'k{idx}')
                
                if obj_type == 'gate':
                    if d[0] not in ['infinite']:
                        description += f"g{idx} is {state_for_llm['gate_status'][idx]}. "
                        if state_for_llm['gate_status'][idx] == 'open':
                            description += f"g{idx} will stay open for {10 - state_for_llm['gate_open_time'][idx]} timesteps. "

                if self.layout_name in ['forced_coordination', 'counter_circuit_o_1order', 'soup_passing']:   
                    if obj_type == 'storage_counter':
                        if state_for_llm['storage_counter_objects'][idx] == 'empty':
                            description += f"s{idx} is empty. "
                        else:
                            description += f"s{idx} contains {state_for_llm['storage_counter_objects'][idx]}. "
                # When there are no kitchen counters:
        if self.enable_kitchen_counters:
            if len(self.empty_kitchen_counter_distances) > 0:
                closest_kitchen_counter = self.empty_kitchen_counter_distances.index(min(self.empty_kitchen_counter_distances))
                distance_to_closest_kitchen_counter = min(self.empty_kitchen_counter_distances)
                # print('Number of kitchen counters: ', len(self.empty_kitchen_counter_distances))
                if distance_to_closest_kitchen_counter != float('inf'):
                    description += f'Closest empty kitchen counter k{closest_kitchen_counter} is {distance_to_closest_kitchen_counter} units away. '

        return description

    def _state_to_description(self, state_for_llm, other_player_message = ''):
#        print('STATE FOR LLM: ', state_for_llm)
        state_for_llm = self._correct_dish_to_plate(state_for_llm)
        description = self._add_history() 
        # Add state information in natural language 
        description += self._add_held_object_info(state_for_llm)
        if not self.single_agent_ablation:
            description += self._add_kitchen_facility_info(state_for_llm)
        else:
            description += self._add_kitchen_facility_info_single_agent_ablation(state_for_llm)


        # get available actions based on current state and add the information to the description
        self.available_actions_list = self._get_available_actions(state_for_llm, None)
        # Uncomment for ToM Reasoning LLM
        # self.partner_inference_string = self.infer_partner_state(description)
        # description += self.partner_inference_string
        available_actions = ""
        for i, action in enumerate(self.available_actions_list):
            available_actions += f'{chr(65 + i)}. {action}\n'
        description += f"\nAvailable Actions:\n{available_actions}"

        return description
