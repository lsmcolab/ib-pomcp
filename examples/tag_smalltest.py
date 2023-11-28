###
# Imports
###
import sys
import os
import time

sys.path.append(os.getcwd())

from src.log import LogFile
from src.utils.args import get_args
from src.envs.TagEnv import load_default_scenario

###
# Setting the environment
###
args = get_args()
env, scenario_id = load_default_scenario(args.atype,scenario_id=args.id,display=True)

###
# ADLEAP-MAS MAIN ROUTINE
###
state = env.reset()
agent = env.components['robot']

header = ['Iteration','Reward','Time to reason','N Rollouts', 'N Simulations']
log = LogFile('TagEnv',scenario_id,args.atype,args.exp_num,header)

MAX_EPISODES = 200
done = False
while not done and env.episode < MAX_EPISODES:
    print(env.episode)
    # 1. Importing agent method
    method = env.import_method(agent.type)

    # 2. Reasoning about next action and target
    start = time.time()
    agent.next_action, _ = method(state, agent)
    end = time.time()
    obs = env.get_observation()
    
    # 3. Taking a step in the environment
    state,reward,done,info = env.step(agent.next_action)
    data = {'it':env.episode,
            'reward':reward,
            'time':end-start,
            'nrollout':agent.smart_parameters['count']['nrollouts'],
            'nsimulation':agent.smart_parameters['count']['nsimulations']}
    log.write(data)
env.close()
###
# THE END - That's all folks :)
###