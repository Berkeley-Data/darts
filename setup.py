from setuptools import setup
setup(
    name='darts',
    version='0.0.1',
    description='Dynamic and responsive targeting system using multi-arm bandits modified for delayed feedback.',
    py_modules=['bandit','allocation'],
    package_dir={'': 'src'}
)