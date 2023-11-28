from src.reasoning.estimation import type_parameter_estimation
from src.reasoning.node import RhoANode, RhoONode, particle_revigoration
import numpy as np
import random
import time

class RhoPOMCP(object):

    def __init__(self,max_depth,max_it,kwargs):
        ###
        # Traditional Monte-Carlo Tree Search parameters
        ###
        self.root = None
        self.episode = 0
        self.max_depth = max_depth
        self.max_it = max_it
        self.c = 0.5
        discount_factor = kwargs.get('discount_factor')
        self.discount_factor = discount_factor\
            if discount_factor is not None else 0.95

        ###
        # POMCP enhancements
        ###
        k = kwargs.get('k') # particle filter size
        self.k = k if k is not None else 100
        
        smallbag_size = kwargs.get('smallbag_size') # smallbag size
        self.smallbag_size = smallbag_size if smallbag_size is not None else 10
        
        time_budget = kwargs.get('time_budget') # time budget in seconds
        self.time_budget = time_budget if time_budget is not None else np.inf
        self.start_time_budget = time.time()
              
        ###
        # Further settings
        ###
        target = kwargs.get('target')
        if target is not None:
            self.target = target
            self.initial_target = target
        else: #default
            self.target = 'max'
            self.initial_target = 'max'
            
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

    def rhofunction(self,particles,action):
        belief_reward = 0.0

        start_t = time.time()
        norm = 0
        if isinstance(particles,dict):
            # calculating belief 
            for key in particles:
                tmp_state = particles[key][0].copy()
                state,reward, _, _ = tmp_state.step(action)
                
                trans_p = (state.get_trans_p(action))[1]
                obs_p = (state.get_obs_p(action))[1]
                belief_reward += reward*trans_p*obs_p*particles[key][1]
                norm += particles[key][1]
        else:
            # calculating belief 
            for particle in particles:
                tmp_state = particle.copy()
                state,reward, _, _ = tmp_state.step(action)
                
                trans_p = (state.get_trans_p(action))[1]
                obs_p = (state.get_obs_p(action))[1]
                belief_reward += reward*trans_p*obs_p
                norm += 1
        end_t = time.time()
        return belief_reward/self.smallbag_size

    def importance_sampling(self,smallbag,action,next_state):
        next_smallbag = []
        next_smallbag.append(next_state)

        start_t = time.time()
        while len(next_smallbag) < self.smallbag_size:
            # (1) sampling the particle from smallbag
            particle = random.choice(smallbag)

            # (2) generating particle' from particle using G
            tmp_state = particle.copy()
            state,reward, _, _ = tmp_state.step(action)

            # (3) storing the generated particle particle' in the new smallbag
            next_smallbag.append(state)
        end_t = time.time()
        
        return next_smallbag
    
    def change_paradigm(self):
        if self.target == 'max':
            return 'min'
        elif self.target == 'min':
            return 'max'
        else:
            raise NotImplemented

    def simulate_action(self, node, action):
        # 1. Copying the current state for simulation
        tmp_state = node.state.copy()

        # 2. Acting
        next_state,reward, _, _ = tmp_state.step(action)
        next_node = RhoANode(action,next_state,node.depth+1,node)

        # 3. Returning the next node and the reward
        return next_node, reward

    def rollout_policy(self,state):
        if getattr(state,'default_policy',None) is not None:
            return state.default_policy()
        return random.choice(state.get_actions_list())

    def rollout(self,node,smallbag):
        # 1. Checking if it is an end state or leaf node
        if self.is_terminal(node) or self.is_leaf(node):
            return 0

        self.rollout_count += 1
        start_t = time.time()

        # 2. Choosing an action
        action = self.rollout_policy(node.state)

        # 3. Simulating the action
        next_state, reward, _, _ = node.state.step(action)
        node.state = next_state
        node.observation = next_state.get_observation()
        node.depth += 2

        next_smallbag = self.importance_sampling(smallbag,action,node.state)

        end_t = time.time()
        self.rollout_total_time += (end_t - start_t)

        # 4. Rolling out
        return self.rhofunction(smallbag, action) +\
            self.discount_factor*self.rollout(node,next_smallbag)

    def get_rollout_node(self,node):
        obs = node.state.get_observation()
        tmp_state = node.state.copy()
        depth = node.depth
        return RhoONode(observation=obs,state=tmp_state,depth=depth,parent=None)

    def is_leaf(self, node):
        if node.depth >= self.max_depth + 1:
            return True
        return False

    def is_terminal(self, node):
        if (time.time() - self.start_time_budget) > self.time_budget:
            return True
        return node.state.state_set.is_final_state(node.state)

    def simulate(self, node, smallbag):
        # 1. Checking the stop condition
        if node.depth == 0:
            node.visits += 1

        if self.is_terminal(node) or self.is_leaf(node):
            return 0

        # 2. Checking child nodes
        if node.children == []:
            # a. adding the children
            for action in node.actions:
                (next_node, reward) = self.simulate_action(node, action)
                node.children.append(next_node)
            rollout_node = self.get_rollout_node(node)
            return self.rollout(rollout_node,smallbag)

        self.simulation_count += 1
        start_t = time.time()

        # 3. Selecting the best action
        action = node.select_action(coef=self.c,mode=self.target)
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
                observation_node.particle_filter.append(action_node.state)
                break
        
        if observation_node is None:
            observation_node = action_node.add_child(observation)
            observation_node.particle_filter.append(observation_node.state)
        observation_node.visits += 1

        # 7. Generating the new smallbag
        next_smallbag = self.importance_sampling(smallbag,action,observation_node.state)
        for particle in smallbag:
            node.particle_filter.append(particle)
            node.add_to_cummulative_bag(particle,action)
        node.particle_filter.append(node.state)
        node.add_to_cummulative_bag(node.state,action)

        end_t = time.time()
        self.simulation_total_time += (end_t - start_t)

        # 8. Calculating the reward, quality and updating the node
        R = self.rhofunction(node.cummulative_bag, action) + \
            float(self.discount_factor * self.simulate(observation_node,next_smallbag))
        node.update(action, R)
        return R

    def search(self, node, agent):
        # 1. Performing the Monte-Carlo Tree Search
        it = 0
        self.start_time_budget = time.time()
        while it < self.max_it:
            self.target = self.initial_target
            
            # a. Sampling the belief state for simulation
            if len(node.particle_filter) < 1 + self.smallbag_size:
                sampled_states = node.state.sample_nstate(agent, 1 + self.smallbag_size)
                beliefState, smallbag = sampled_states[0], sampled_states[1:]
            else:
                sampled_states = random.sample(node.particle_filter, 1 + self.smallbag_size)
                beliefState, smallbag = sampled_states[0], sampled_states[1:]
            node.state = beliefState

            # b. simulating
            self.simulate(node,smallbag)
            
            if (time.time() - self.start_time_budget) > self.time_budget:
                return node.get_best_action()
            it += 1

        return node.get_best_action()

    def planning(self, state, agent):
        # 1. Getting the current state and previous action-observation pair
        previous_action = agent.next_action
        current_observation = state.get_observation()

        # 2. Defining the root of our search tree
        # via initialising the tree
        if self.root is None:
            self.root = RhoONode(observation=None,state=state,depth=0,parent=None)
        # or advancing within the existent tree
        else:
            self.root = find_new_PO_root(state, previous_action,\
             current_observation, agent, self.root, adversary=self.adversary)

        # 3. Estimating the parameters 
        if 'estimation_method' in agent.smart_parameters:
            self.root.state, agent.smart_parameters['estimation'] = \
             type_parameter_estimation(self.root.state,agent, agent.smart_parameters\
              ['estimation_method'], *agent.smart_parameters['estimation_args'])

        # 4. Performing particle revigoration
        particle_revigoration(state,agent,self.root,self.k)

        # 5. Searching for the best action within the tree
        best_action = self.search(self.root, agent)

        # 6. Returning the best action
        self.root.show_qtable()
        info = { 'nrollouts': self.rollout_count,
            'nsimulations':self.simulation_count}
        return best_action, info

def rhopomcp_planning(env, agent, max_depth=20, max_it=250, **kwargs):    
    # 1. Setting the environment for simulation
    copy_env = env.copy()
    copy_env.simulation = True

    # 2. Planning
    rhopomcp = RhoPOMCP(max_depth, max_it, kwargs) if 'rhopomcp' not \
     in agent.smart_parameters else agent.smart_parameters['rhopomcp']
     
    # - planning
    next_action, info = rhopomcp.planning(copy_env,agent)

    # 3. Updating the search tree
    agent.smart_parameters['rhopomcp'] = rhopomcp
    agent.smart_parameters['count'] = info
    return next_action,None

def find_new_PO_root(current_state, previous_action, current_observation, 
 agent, previous_root, adversary=False):
    # 1. If the root doesn't exist yet, create it
    # - NOTE: The root is always represented as an "observation node" since the 
    # next node must be an action node.
    if previous_root is None:
        new_root = RhoONode(observation=None,state=current_state,depth=0,parent=None)
        return new_root

    # 2. Else, walk on the tree to find the new one (giving the previous information)
    action_node, observation_node, new_root = None, None, None

    # a. walking over action nodes
    for child in previous_root.children:
        if child.action == previous_action:
            action_node = child
            break

    # - if we didn't find the action node, create a new root
    if action_node is None:
        new_root = RhoONode(observation=None,state=current_state,depth=0,parent=None)
        return new_root

    # b. walking over observation nodes
    for child in action_node.children:
        if child.state.observation_is_equal(current_observation):
            observation_node = child
            break

    # - if we didn't find the action node, create a new root
    if observation_node is None:
        new_root = RhoONode(observation=None,state=current_state,depth=0,parent=None)
        return new_root

    # c. checking the adversary condition
    if adversary:
        action_node, observation_node = None, None
        for child in new_root.children:
            if child.action == agent.smart_parameters['adversary_last_action']:
                action_node = child
                break
        # - if we didn't find the action node, create a new root
        if action_node is None:
            new_root = RhoONode(\
                observation=None,state=current_state,depth=0,parent=None)
            return new_root

        for child in action_node.children:
            if child.state.observation_is_equal(\
             agent.smart_parameters['adversary_last_observation']):
                observation_node = child
                break
        # - if we didn't find the action node, create a new root
        if observation_node is None:
            new_root = RhoONode(\
                observation=None,state=current_state,depth=0,parent=None)
            return new_root

    # 3. Definig the new root and updating the depth
    new_root = observation_node
    new_root.parent = None
    new_root.update_depth(0)
    return new_root