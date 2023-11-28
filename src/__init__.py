from gymnasium.envs.registration import register

register(
    id='AdhocReasoningEnv-v1',
    entry_point='src.envs:AdhocReasoningEnv',
)

register(
    id='TigerEnv-v2',
    entry_point='src.envs:TigerEnv',
    )

register(
    id='MazeEnv-v2',
    entry_point='src.envs:MazeEnv',
)

register(
    id='RockSampleEnv-v2',
    entry_point='src.envs:RockSampleEnv',
)

register(
    id='TagEnv-v1',
    entry_point='src.envs:TagEnv',
)

register(
    id='LaserTagEnv-v1',
    entry_point='src.envs:TagEnv',
)

register(
    id='LevelForagingEnv-v2',
    entry_point='src.envs:LevelForagingEnv',
)