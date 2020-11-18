from setuptools import setup

with open('README.md','r') as fh:
    long_description = fh.read()

setup(
    name='darts-berkeley',
    version='0.0.2',
    description='Dynamic and responsive targeting system using multi-arm bandits modified for delayed feedback.',
    py_modules=['bandit','allocation'],
    package_dir={'': 'src'},
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Natural Language :: English',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Berkeley-Data/darts',
    author='Nick Sylva',
    author_email='nick.sylva@berkeley.edu'
)