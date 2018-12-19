from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='imbalanced_databases',
      version='0.1',
      description='imbalanced databases',
      long_description=readme(),
      classifiers=[
              'Development Status :: 3 - Alpha',
              'License :: OSI Approved :: MIT License',
              'Programming Language :: Python',
              'Topic :: Scientific/Engineering :: Artificial Intelligence'],
      url='http://github.com/gykovacs/imbalanced_databases',
      author='Gyorgy Kovacs',
      author_email='gyuriofkovacs@gmail.com',
      license='MIT',
      packages=['imbalanced_databases'],
      install_requires=[
              'numpy',
              'pandas',
              'scipy',
              'sklearn',
              ],
      py_modules=['imbalanced_databases'],
      zip_safe=False,
      package_dir= {'imbalanced_databases': 'imbalanced_databases'},
      package_data= {'imbalanced_databases': ['data/*/*']},
      tests_require= ['nose'],
      test_suite= 'nose.collector'
      )
