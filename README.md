# Information-guided Planning: An Online Approach for Partially Observable Problems

<i>In Proceedings of the 37th Conference on Neural Information Processing Systems. 2023.</i> <a href="#alves2023information">[1]</a>

If you use our solutions as one of your baselines, please cite us!

```
@inproceedings{alves2023information,
  author = {do Carmo Alves, Matheus Aparecido and Varma, Amokh and Elkhatib, Yehia and Soriano Marcolino, Leandro},
  title = {Information-guided Planning: An Online Approach for Partially Observable Problems},
  year = {2023},
  isbn = {},
  address = {New Orleans Ernest N. Morial Convention Center},
  abstract = {This paper presents IB-POMCP, a novel algorithm for online planning under partial observability. Our approach enhances the decision-making process by using estimations of the world belief's entropy to guide a tree search process and surpass the limitations of planning in scenarios with sparse reward configurations. By performing what we denominate as an information-guided planning process, the algorithm, which incorporates a novel I-UCB function, shows significant improvements in reward and reasoning time compared to state-of-the-art baselines in several benchmark scenarios, along with theoretical convergence guarantees.},
  booktitle = {Proceedings of the 37th Conference on Neural Information Processing Systems},
  numpages = {21},
  series = {NeurIPS 2023}
}
```

## WHAT IS IB-POMCP? :open_mouth:

<p style="text-align: justify; text-indent: 10px;" >
<i>IB-POMCP</i> is a novel algorithm for online planning under partial observability. Our proposal presents a fresh approach that enhances the decision-making process of autonomous agents by using the estimations of the world belief's entropy to guide a tree search process and surpass the limitations of planning in scenarios with sparse reward configurations. By performing what we denominate as an information-guided planning process, the algorithm, which incorporates a novel I-UCB function, shows significant improvements in reward and reasoning time compared to state-of-the-art baselines in several benchmark scenarios, along with theoretical convergence guarantees. More detail and information about our approach can be found in <a href="#alves2023information">our paper [1]</a>.
</p>

        
## SUMMARY

In this README you can find:

- [Information-guided Planning: An Online Approach for Partially Observable Problems](#information-guided-planning-an-online-approach-for-partially-observable-problems)
  - [WHAT IS IB-POMCP? :open\_mouth:](#what-is-ib-pomcp-open_mouth)
  - [SUMMARY](#summary)
  - [GET STARTED](#get-started)
    - [1. Dependencies :pencil:](#1-dependencies-pencil)
    - [2. Usage :muscle:](#2-usage-muscle)
    - [3. IB-POMCP code details :computer:](#3-ib-pomcp-code-details-computer)
  - [REFERENCES](#references)

## GET STARTED

### 1. Dependencies :pencil: 

<b>- About this repository</b>

This repository represents a streamlined version of the environment used during our research and proposal of IB-POMCP. 
We removed some files and improved comments in order to facilitate your reading and understanding through the code. :smile:

As mentioned in our paper, we utilized the <a href="#alves2022adleapmas"><i>AdLeap-MAS</i> framework [2]</a> to conduct all experiments and analyze the results. Therefore, the dependencies outlined here mirror those of the framework; however, we provide the minimal set required to run IB-POMCP code, the baselines and the benchmarks presented in the paper, double-check `requirements.txt`.

<b>- Encountering issues while running our code?</b> :fearful: 

<p style="text-align: justify; text-indent: 0px;">
 If you find yourself unable to resolve them using our tutorial, we recommend consulting the <a href="https://github.com/lsmcolab/adleap-mas/">AdLeap-MAS GitHub page</a> for additional guidance on troubleshooting common problems or contact us here on GitHub!
</p>

------------------------
### 2. Usage :muscle:

<b>- Quick experience</b>

For a quick experience, we recommend running the default `main.py` file, which will run a IB-POMCP's experiment in the Foraging environment, U-Shaped scenario. By default, the display will pop-up for visual evaluation of the agent's behaviour and a result file will be created in `results/` folder, which can be directly used in plots later.

<b>- Running different environments and baselines</b>

If you want to run your experiment in other environments, you will find some options at the top of the `main.py` file.

```python
# 1. Setting the environment
method = 'ibpomcp'              # choose your method
kwargs = {}                     # define your additional hyperparameters to it (optional)
env_name = 'LevelForagingEnv'   # choose your environment
scenario_id = 2                 # define your scenario configuration (check the available configuration in our GitHub)

display = True                  # choosing to turn on or off the display
```

Directly, you can change the environment by modifying the `env_name` and the `scenario_id` variables there.
We have 6 different options for `env_name`, which refer to the environments in `src/envs/` folder.
Each environment will present a different range as options for `scenarios_id`. 
To help you out in this task, we list down all the possibilities:

```python
...
env_name = 'LevelForagingEnv' # Tiger, MazeEnv, RockSampleEnv, TagEnv, LaserTagEnv
...
scenario_id = 2 # Tiger: [0] 
                # LevelForagingEnv: [0,1,2,3,4]
                # MazeEnv: [0,1,2,3]
                # RockSampleEnv: [0,1,2,3]
                # TagEnv: [0]
                # LaserTagEnv: [0]
```

On the other hand, if you want to run a different baseline in some of these scenarios, you just need to change the `method` variable!
Straightforwardly, we have 5 options: IB-POMCP, POMCP, rho-POMCP, IPR-POMCP and IUCB-POMCP.

```python
method = 'ibpomcp' # ibpomcp, pomcp, rhopomcp, iprpomcp, iucbpomcp
```

But wait, you might ask yourself (if you read the paper): where is TB rho-POMCP?
No worries, I gotchu! 
TB rho-POMCP is modified version of rho-POMCP, hence, if you want to run it you just need to modify an internal argument of rho-POMCP in `src/reasoning/rhopomcp.py:31` file.

```python
...
time_budget = kwargs.get('time_budget') # time budget in seconds
self.time_budget = time_budget if time_budget is not None else np.inf
...
```

If you want, you can pass the time-budget value as an argument while initialising the algorithm or directly modify it inside the rho-POMCP code.
To set it as an argument, you can use the `kwargs` option in the main file, something like:

```python
... 
kwargs = {'time_budget':5} # 5 sec of time-budget
...
```

That's it folks. Easy and ready to use. :innocent:

------------------------
### 3. IB-POMCP code details :computer:

Let us now briefly present IB-POMCP's code. In this section, we aim to facilitate your understanding and offer some guidance through some key points of the algorithm, which details can be found in our paper and match our code.

Besides that, IB-POMCP is an algorithm based on <a href="#silver2010"><i>POMCP</i> [3]</a>, hence, while passing through its code you might find some similarities to the algorithm provided in David Silver's POMCP paper.

Our algorithm is implemented in `src/reasoning/ibpomcp.py` and it starts in line 262 with:

```python
262 | def ibpomcp_planning(env, agent, max_depth=20, max_it=250, **kwargs):
... |    ...
```

<p style="text-align: justify; text-indent: 0px;">
From there, the IB-POMCP planning will follow a similar search strategy to POMCP.
That is, the planning procedure starts with the initialisation or update of the root node (line 240) and the performance of the particle reinvigoration process to boost the search process (line 251). Subsequently, the search process starts (line 206), where we sample (line 214) and simulate (line 221) a particle/state to evaluate and reason the world possibilities. The simulation responsibility is to expand the tree by selecting actions (line 159) and estimating rewards for them in the game (line 296). The process is recursive and finishes after performing a fixed number of iterations.
In code, you can observe this high-level process as:
</p>

```python
IBPOMCP
...
138 | def simulate(self, node):
139 |    # 1. Checking the stop condition
    |    ...
143 |    if self.is_terminal(node) or self.is_leaf(node):
144 |       return 0, [node.state]
    |    ...
146 |    # 2. Checking child nodes
147 |    if node.children == []:
148 |       # a. adding the children
    |       ...
153 |       return self.rollout(rollout_node), [node.state]
    |    ...
158 |    # 3. Selecting the best action
159 |    action = node.select_action(coef=self.alpha,mode=self.target)
160 |    self.target = self.change_paradigm() if self.adversary else self.target   
    |    ...
162 |    # 4. Simulating the action
163 |    (action_node, reward) = self.simulate_action(node, action)
    |    ...
165 |    # 5. Adding the action child on the tree
    |    ...
176 |    # 6. Getting the observation and adding the observation child on the tree
    |    ...
194 |    # 7. Calculating the reward, quality and updating the node
295 |    future_reward, observation_states_found = self.simulate(observation_node)
296 |    R = reward + (self.discount_factor * future_reward)
297 |
298 |    # - node update
299 |    node.add_to_observation_distribution(observation_states_found)
200 |    node.particle_filter.append(node.state)
201 |    node.update(action, R)
202 |
203 |    observation_states_found.append(node.state)
204 |    return R, observation_states_found

--------------------

206 | def search(self, node, agent):
207 |     # 1. Performing the Monte-Carlo Tree Search
208 |     it = 0
209 |     while it < self.max_it:
    |         ...
212 |         # a. Sampling the belief state for simulation
214 |             beliefState = node.state.sample_state(agent)
    |         ...
219 |         # b. simulating
220 |         self.alpha = node.get_alpha()
221 |         self.simulate(node)
222 |         it += 1
    | ...
226 | return node.get_best_action(self.alpha,self.target)

--------------------

228 | def planning(self, state, agent):
    |    ...
233 |    # 2. Defining the root of our search tree
234 |    # via initialising the tree
    |    ...
240 |        self.root, Px = find_new_PO_root(state, previous_action,\
241 |         current_observation, agent, self.root, adversary=self.adversary)
    |    ...
249 |    # 4. Performing particle revigoration
250 |    if self.pr:
251 |       particle_revigoration(state,agent,self.root,self.k, Px)
    |    ...
253 |    # 5. Searching for the best action within the tree
254 |    best_action = self.search(self.root, agent)
    |    ...
260 |    return best_action, info

--------------------

262 | def ibpomcp_planning(env, agent, max_depth=20, max_it=250, **kwargs):
    |    ...
278 |    return next_action,None
...
```

When reading the paper, you can use this scheme (together the actual code) to double-check and match what we discuss in the paper with our implemention details. :blush:

Last but not the least, all functions are implemented and available in our code (obviously) and can be found at:

<b>Alpha function</b> `src/reasoning/node.py:339`

```python
    def get_alpha(self):
        adjust_value = 0.2
        if self.visits == 0:
            return 1 - adjust_value

        decaying_factor = math.e*math.log(self.visits)/self.visits
        entropy_value_trend = self.cumentropy/\
                            (self.visits*self.max_entropy)
        norm_entropy = decaying_factor*entropy_value_trend

        return (1 - 2*adjust_value)*(norm_entropy) +adjust_value
```

<b>I-UCB action selection</b> `src/reasoning/qlearn.py:81`

```python
    def iucb_select_action(node,alpha,mode='max'):
        ...
        information_value = node.etable[str(a)]['entropy']/\
                                node.etable[str(a)]['max_entropy']

        current_ucb =  qvalue + \
            ((1-alpha) * exploration_value) + (alpha * information_value)
        ...
```

<b>Entropy Update function</b> `src/reasoning/node.py:352`

```python
    def update(self, action, result):
        ...
        result_entropy = entropy(self.observation_distribution)
        self.etable[str(action)]['trials'] += 1

        self.etable[str(action)]['cumentropy'] += result_entropy
        self.cumentropy += result_entropy

        self.etable[str(action)]['entropy'] += \
            (float(result_entropy) - self.etable[str(action)]['entropy']) /\
                 float(self.etable[str(action)]['trials'])
        self.entropy += (result_entropy - self.entropy) / self.visits   
        ...
```

That's it folks. Hope you can find it useful for you. :innocent:

------------------------
## REFERENCES

<a name="alves2023information">[1]</a> Matheus Aparecido do Carmo Alves, Amokh Varma, Yehia Elkhatib, and Leandro Soriano Marcolino. 2023. <b>Information-guided Planning: An Online Approach for Partially Observable Problems</b>. In Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS 2023). New Orleans Ernest N. Morial Convention Center.

<a name="alves2022adleapmas">[2]</a> Matheus Aparecido do Carmo Alves, Amokh Varma, Yehia Elkhatib, and Leandro Soriano Marcolino. 2022. <b>AdLeap-MAS: An Open-source Multi-Agent Simulator for Ad-hoc Reasoning</b>. In Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems (AAMAS '22). International Foundation for Autonomous Agents and Multiagent Systems, Richland, SC, 1893–1895.

<a name="silver2010">[3]</a> David Silver and Joel Veness. 2010. <b>Monte-Carlo planning in large POMDPs</b>. In Proceedings of the 23rd International Conference on Neural Information Processing Systems - Volume 2 (NIPS'10). Curran Associates Inc., Red Hook, NY, USA.