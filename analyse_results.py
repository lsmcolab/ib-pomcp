import src.utils.plot as plt
import src.utils.stats as sts
import src.utils.results as rpck
# if warnings are disturbing the presentation, uncomment the lines bellow
import warnings
warnings.filterwarnings("ignore")

# 1. Defining analysis  settings
NEXP = 50
PATH = './results/'
SAVE = True

SHOW_SUMMARY = True
SHOW_PVALUE = True

PLOT = True
PLOT_TYPE = 'cumlines'

# select the target data
target_data = ['reward']#,'time','nrollouts','nsimulations'
ylabel = {
    'lines':{
        'reward':'Average Reward',
        'time':'Average Time (s)'},
    'cumlines':{
        'reward':'Cumulative Reward',
        'time':'Cumulative Time (s)'},
    'bars':{
        'reward':'Average Reward',
        'time':'Average Time (s)'},
    }    

# select the target environments
envs = ['TigerEnv0',
        'MazeEnv0','MazeEnv1','MazeEnv2','MazeEnv3',
        'RockSampleEnv0','RockSampleEnv1','RockSampleEnv2','RockSampleEnv3',
        'LevelForagingEnv0','LevelForagingEnv1','LevelForagingEnv2','LevelForagingEnv3', 'LevelForagingEnv4',
        'TagEnv0','LaserTagEnv0'
        ]

# select the target methods
methods_dict = {
    'pomcp':'POMCP',
    'prpomcp':'PR-POMCP',
    'iucbpomcp':'IUCB-POMCP',
    'ibpomcp':'IB-POMCP',
    'tbrhopomcp':'TB ρ-POMCP',
    'rhopomcp':'ρ-POMCP',
}
methods = [name for name in methods_dict]

for env in envs:
    print('>',env)

    results = {}
    for method in methods:
        results[methods_dict[method]] = \
            rpck.read(nexp=NEXP,method=method,path=PATH,env=env)
        
    # 2. Analysing via plot and pvalues
    for td in target_data:
        if SHOW_SUMMARY:
            sts.summary(results=results,target_data=td,LaTeX=True)

        if SHOW_PVALUE:
            #sts.pvalues(results=results,target_data=td,by_='iteration')
            sts.pvalues(results=results,target_data='reward',by_='experiment')

        if PLOT:
            if PLOT_TYPE == 'lines':
                plt.lines(results=results,target_data=td,ylabel=ylabel[PLOT_TYPE][td],
                    xlabel='Iteration',save=SAVE,savepath='./plots/',env_name=env)

            elif PLOT_TYPE == 'cumlines':
                plt.cumlines(results=results,target_data=td,
                            ylabel=ylabel[PLOT_TYPE][td],xlabel='Iteration',
                            save=SAVE,savepath='./plots/',env_name=env)
            elif PLOT_TYPE == 'bars':
                plt.bars(results=results,target_data=td,ylabel=ylabel[PLOT_TYPE][td],
                            save=SAVE,savepath='./plots/',env_name=env)