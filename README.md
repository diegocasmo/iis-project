# IIS Project

## Installation
  - [Install Anaconda ](https://www.continuum.io/downloads)
  - Run ``conda env create -f environment.yml`` to install dependencies
  - Run ``source activate iis`` to activate environment

### Running ``lm2`` Features Analysis
  - Unzip dataset in the ``./data`` directory
    - Make sure all subdirectories are also unzipped
    - Do not change any of the original file names nor original directory structure
  - From the root directory, run ``python code/lm2-analysis/lm2_parser.py`` to create the ``./data/lm2.csv`` file
  - Once the conda environment has been activated, run ``jupyter notebook`` in your terminal
  - Open browser at http://localhost:8888/?token=<your_token>
  - Navigate to the ``code/lm2-analysis/solution.ipynb`` file
