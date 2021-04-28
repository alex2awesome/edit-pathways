from setuptools import setup, find_packages

setup(
    name='get_edit_diffs',
    version='0.0.1',
    description='Calculate bipartite graph of sentence-matches',
    url='https://bbgithub.dev.bloomberg.com/aspangher/edit-project',
    author='Alex Spangher',
    author_email='aspangher@bloomberg.net',
    license='',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().split('\n'),
    entry_points={
      'console_scripts': [
          'run_pyspark.py = spark.runner_script:main'
      ]
    },
    zip_safe=False
)