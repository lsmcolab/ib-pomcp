from setuptools import setup

#####
# Base Environment
#####
setup(name='AdhocReasoningEnv',
      version='1.0.0',
      install_requires=['gym']
)

#####
# Paper Benchmark Environments
#####
setup(name='TigerEnv',
      version='2.0.0',
      install_requires=['gym','numpy'],
)

setup(name='MazeEnv',
      version='2.0.0',
      install_requires=['gym','numpy']
)

setup(name='RockSampleEnv',
      version='2.0.0',
      install_requires=['gym','numpy']
)

setup(name='TagEnv',
      version='1.0.0',
      install_requires=['gym','numpy']
)

setup(name='LaserTagEnv',
      version='1.0.0',
      install_requires=['gym','numpy']
)

setup(name='LevelForagingEnv',
      version='2.0.0',
      install_requires=['gym','numpy']
)