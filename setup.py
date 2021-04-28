from setuptools import setup

setup(
    name='get_edit_diffs',
    version='0.0.1',
    description='Calculate bipartite graph of sentence-matches',
    url='https://bbgithub.dev.bloomberg.com/aspangher/edit-project',
    author='Alex Spangher',
    author_email='aspangher@bloomberg.net',
    license='',
    packages=['extract_newsarchive'],
    install_requires=open('requirements.txt').split('\n'),
    entry_points={
      'console_scripts': [
          'run_pyspark.py = spark.runner_script:main'
      ]
    },
    zip_safe=False
)