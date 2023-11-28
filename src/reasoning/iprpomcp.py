from src.reasoning.node import IANode, IONode
from src.reasoning.qlearn import entropy

import numpy as np
import random
import time
from src.reasoning.estimation import type_parameter_estimation

class PRPOMCP(object):

    def __init__(self,max_depth,max_it,kwargs):
        ###
        # Traditional Monte-Carlo Tree Search parameters
        ###
        self.root = None
        self.max_depth = max_depth
        self.max_it = max_it
        
        discount_factor = kwargs.get('discount_factor')
        self.discount_factor = discount_factor\
            if discount_factor is not None else 0.95

        self.alpha = 0.5

        ###
        # POMCP enhancements
        ###
        # particle Revigoration (silver2010pomcp)
        particle_revigoration = kwargs.get('particle_revigoration')
        if particle_revigoration is not None:
            self.pr = particle_revigoration
        else: #default
            self.pr = True

        k = kwargs.get('k') # particle filter size
        self.k = k if k is not None else 100

        ###
        # Further settings
        ###
        target = kwargs.get('target')
        if target is not None:
            self.target = target
            self.initial_target = target
        else: #default
            self.target = 'iucb-max'
            self.initial_target = 'iucb-max'
            
        adversary_mode = kwargs.get('adversary')
        if adversary_mode is not None:
            self.adversary = adversary_mode
        else: #default
            self.adversary = False
            
        stack_size = kwargs.get('state_stack_size')
        if stack_size is not None:
            self.state_stack_size = stack_size
        else: #default
            self.state_stack_size = 1

        ###
        # Evaluation
        ###
        self.rollout_total_time = 0.0
        self.rollout_count = 0.0
        
        self.simulation_total_time = 0.0
        self.simulation_count = 0.0

    def change_paradigm(self):
        if self.target == 'iucb-max':
            return 'iucb-min'
        elif self.target == 'iucb-min':
            return 'iucb-max'
        else:
            raise NotImplemented

    def simulate_action(self, node, action):
        # 1. Copying the current state for simulation
        tmp_state = node.state.copy()

        # 2. Acting
        next_state,reward, _, _ = tmp_state.step(action)
        next_node = IANode(action,next_state,node.depth+1,node)

        # 3. Returning the next node and the reward
        return next_node, reward

    def rollout_policy(self,state):
        return random.choice(state.get_actions_list())

    def rollout(self,node):
        # 1. Checking if it is an end state or leaf node
        if self.is_terminal(node) or self.is_leaf(node):
            return 0, []

        self.rollout_count += 1
        start_t = time.time()

        # 2. Choosing an action
        action = self.rollout_policy(node.state)

        # 3. Simulating the action
        next_state, reward, _, _ = node.state.step(action)
        node.state = next_state
        node.observation = next_state.get_observation()
        node.depth += 2

        end_t = time.time()
        self.rollout_total_time += (end_t - start_t)

        # 4. Rolling out
        future_reward, _ = self.rollout(node)
        R = reward + (self.discount_factor * future_reward)
        return R, []

    def get_rollout_node(self,node):
        obs = node.state.get_observation()
        tmp_state = node.state.copy()
        depth = node.depth
        return IONode(observation=obs,state=tmp_state,depth=depth,parent=None)

    def is_leaf(self, node):
        if node.depth >= self.max_depth + 1:
            return True
        return False

    def is_terminal(self, node):
        return node.state.state_set.is_final_state(node.state)

    def simulate(self, node):
        # 1. Checking the stop condition
        if node.depth == 0:
            node.visits += 1

        if self.is_terminal(node) or self.is_leaf(node):
            return 0, [node.state]

        # 2. Checking child nodes
        if node.children == []:
            # a. adding the children
            for action in node.actions:
                (next_node, reward) = self.simulate_action(node, action)
                node.children.append(next_node)
            rollout_node = self.get_rollout_node(node)
            return self.rollout(rollout_node)

        self.simulation_count += 1
        start_t = time.time()
        
        # 3. Selecting the best action
        action = node.select_action(coef=0.5,mode='max')
        self.target = self.change_paradigm() if self.adversary else self.target   

        # 4. Simulating the action
        (action_node, reward) = self.simulate_action(node, action)

        # 5. Adding the action child on the tree
        if action_node.action in [c.action for c in node.children]:
            for child in node.children:
                if action_node.action == child.action:
                    child.state = action_node.state.copy()
                    action_node = child
                    break
        else:
            node.children.append(action_node)
        action_node.visits += 1

        # 6. Getting the observation and adding the observation child on the tree
        observation_node = None
        observation = action_node.state.get_observation()

        for child in action_node.children:
            if child.observation == observation:
                observation_node = child
                observation_node.state = action_node.state.copy()
                break
        
        if observation_node is None:
            observation_node = action_node.add_child(observation)
            observation_node
        observation_node.visits += 1

        end_t = time.time()
        self.simulation_total_time += (end_t - start_t)

        # 7. Calculating the reward, quality and updating the node
        future_reward, observation_states_found = self.simulate(observation_node)
        R = reward + (self.discount_factor * future_reward)

        # - node update
        node.particle_filter.append(node.state)
        node.update(action, R)

        observation_states_found.append(node.state)
        return R, observation_states_found

    def search(self, node, agent):
        # 1. Performing the Monte-Carlo Tree Search
        it = 0
        while it < self.max_it:
            self.target = self.initial_target # reseting the optimisation taret
            
            # a. Sampling the belief state for simulation
            if len(node.particle_filter) == 0:
                beliefState = node.state.sample_state(agent)
            else:
                beliefState = random.sample(node.particle_filter,1)[0]
            node.state = beliefState

            # b. simulating
            self.simulate(node)
            it += 1

        self.target = self.initial_target
        return node.get_best_action(0.0,self.target)

    def planning(self, state, agent):
        # 1. Getting the current state and previous action-observation pair
        previous_action = agent.next_action
        current_observation = state.get_observation()

        # 2. Defining the root of our search tree
        # via initialising the tree
        if self.root is None:
            Px = 0
            self.root = IONode(observation=None,state=state,depth=0,parent=None)
        # or advancing within the existent tree
        else:
            self.root, Px = find_new_PO_root(state, previous_action,\
             current_observation, agent, self.root, adversary=self.adversary)
        
        # 3. Estimating the parameters 
        if 'estimation_method' in agent.smart_parameters:
            self.root.state, agent.smart_parameters['estimation'] = \
             type_parameter_estimation(self.root.state,agent, agent.smart_parameters\
              ['estimation_method'], *agent.smart_parameters['estimation_args'])

        # 4. Performing particle revigoration
        if self.pr:
            particle_revigoration(state,agent,self.root,self.k, Px)

        # 5. Searching for the best action within the tree
        best_action = self.search(self.root, agent)

        # 6. Returning the best action
        self.root.show_qtable()
        info = { 'nrollouts': self.rollout_count,
            'nsimulations':self.simulation_count}
        return best_action, info

def prpomcp_planning(env, agent, max_depth=20, max_it=250, **kwargs):    
    # 1. Setting the environment for simulation
    copy_env = env.copy()
    copy_env.simulation = True

    # 2. POMCP Planning
    # - initialising/getting the plannin algorithm
    prpomcp = PRPOMCP(max_depth, max_it, kwargs) if 'prpomcp' not \
     in agent.smart_parameters else agent.smart_parameters['prpomcp']
    
    # - planning
    next_action, info = prpomcp.planning(copy_env,agent)

    # 3. Updating the search tree
    agent.smart_parameters['prpomcp'] = prpomcp
    agent.smart_parameters['count'] = info
    return next_action,None

###
# POMCP's proposed modification 
###
# POMCP uses find_new_PO_root from node.py module
# > from src.reasoning.node import find_new_PO_root
def find_new_PO_root(current_state, previous_action, current_observation, 
 agent, previous_root, adversary=False):
    # 1. If the root doesn't exist yet, create it
    # - NOTE: The root is always represented as an "observation node" since the 
    # next node must be an action node.
    Px = 0
    if previous_root is None:
        new_root = IONode(observation=None,state=current_state,depth=0,parent=None)
        return new_root, Px

    # 2. Else, walk on the tree to find the new one (giving the previous information)
    action_node, observation_node, new_root = None, None, None

    # a. walking over action nodes
    for child in previous_root.children:
        if child.action == previous_action:
            action_node = child
            break

    # - if we didn't find the action node, create a new root
    if action_node is None:
        new_root = IONode(observation=None,state=current_state,depth=0,parent=None)
        return new_root, Px

    # b. walking over observation nodes
    for child in action_node.children:
        if child.state.observation_is_equal(current_observation):
            observation_node = child
            break

    # - if we didn't find the action node, create a new root
    if observation_node is None:
        new_root = IONode(observation=None,state=current_state,depth=0,parent=None)
        return new_root, Px

    # c. checking the adversary condition
    if adversary:
        action_node, observation_node = None, None
        for child in new_root.children:
            if child.action == agent.smart_parameters['adversary_last_action']:
                action_node = child
                break
        # - if we didn't find the action node, create a new root
        if action_node is None:
            new_root = IONode(\
                observation=None,state=current_state,depth=0,parent=None)
            return new_root, Px

        for child in action_node.children:
            if child.state.observation_is_equal(\
             agent.smart_parameters['adversary_last_observation']):
                observation_node = child
                break
        # - if we didn't find the action node, create a new root
        if observation_node is None:
            new_root = IONode(\
                observation=None,state=current_state,depth=0,parent=None)
            return new_root, Px

    # 3. Definig the new root and updating the depth
    new_root = observation_node
    Px = new_root.visits/previous_root.visits
    new_root.parent = None
    new_root.update_depth(0)
    return new_root, Px

# POMCP uses particle_revigoration from node.py module
# > from src.reasoning.node import particle_revigoration
def particle_revigoration(env,agent,root,k, Px):
    # 1. Copying the current root particle filter
    current_particle_filter = []
    for particle in root.particle_filter:
        current_particle_filter.append(particle)
    Px =  Px if len(current_particle_filter) > 1 else 0.0
    
    # 2. Reinvigorating particles for the new particle filter or
    # picking particles from the uniform distribution
    root.particle_filter = []
    particle_counter = 0
    while(particle_counter < (Px)*k):
        particle = random.sample(current_particle_filter,1)[0]
        root.particle_filter.append(particle)
        particle_counter += 1
        

    particle_counter = 0
    while(particle_counter < (1-Px)*k):
        particle = env.sample_state(agent)
        root.particle_filter.append(particle)
        particle_counter += 1