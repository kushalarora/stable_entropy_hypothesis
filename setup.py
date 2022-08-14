from setuptools import setup, find_packages

with open('requirements.txt') as f:
    reqs = []
    for line in f:
        line = line.strip()
        reqs.append(line.split('==')[0])

setup(
    name='entropy_aware_search',
    version="0.0.1",
    python_requires='>=3.7',
    packages=find_packages(exclude=('data', 'docs', 'logs')),
    install_requires=reqs,
    include_package_data=True,
    package_data={'': ['*.txt', '*.md', '*.opt']},
)
