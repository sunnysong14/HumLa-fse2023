[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8272293.svg)](https://doi.org/10.5281/zenodo.8272293)

*********************************************************************************************
# HumLa: A Practical Human Labeling Method for Online Just-in-Time Software Defect Prediction

This repository contains the source codes together with the datasets to replicate the following paper:

> Liyan Song, Leandro L. Minku, Cong Teng and Xin Yao. "A Practical Human Labeling Method for Online Just-in-Time Software Defect Prediction", in ACM Symposium on the Foundations of Software Engineering (FSE), 2023 (accepted)

## Dependencies
`HumLa and ECo-HumLa` work with Python 3.5+ **only**.

`HumLa and ECo-HumLa` require [numpy](www.numpy.org) to be already installed in your system. 
There are multiple ways to install `numpy`, and the easiest one is
using [pip](https://pip.pypa.io/en/stable/#):
```bash
$ pip install -U numpy
```

[Cython](https://cython.org/) is also required. 
Cython can be installed using pip:
```bash
$ pip install -U Cython
```

Another required package is [skmultiflow](https://scikit-multiflow.readthedocs.io/en/stable/installation.html). The most convenient way to install `skmultiflow` is from [conda-forge](https://anaconda.org/conda-forge/scikit-multiflow) using the following command:
```bash
$  conda install -c conda-forge scikit-multiflow
```
Note: this will install the latest (stable) release. 

Our code was developed with Python 3.9 and relied on the below packages.
- numpy version 1.21.5
- sklearn version 1.0.2
- scikit-multiflow version 0.5.3

But any python environment that satisfies the above requirements should also be workable.

## Structure
To setup the editor (such as `PyCharm`), we recommend to adopt `HumLa2023/` as the root directory for the configuration of directories. 
After setting up the Python environment, the code would be ready to run. 

There are two main folders in the replication package `HumLa2023/` at first, namely `codes/` and `datasets/`:
- The folder `codes/` contains the code scripts implementing the submitted manuscript; 
- The folder `dataset/` contains the 14 datasets produced based on the GitHub open source projects as explained in Section 4 of the submitted paper.

The running the code will then create a third folder `results/` to store ongoing results.


### Core Scripts
The `code/` folder, as it is named, contains the core scripts that implement our RQs and the proposed HumLa and Eco-HumLa. The important two scripts are explained as follows.

- `run_RQs.py` implements RQ1 (1.1-1.3) and RQ2 (2.1-2.3). 
  - `RQ1()` implements RQ1.1 - RQ1.3;
  - `RQ2()` implements RQ2.1 - RQ2.3.

- `main.py` contains several important methods implementing HumLa and Eco-HumLa. It also contains methods that compute the cumulative code churn of HumLa (Eco-HumLa).
  - `run_HumLa()` implements the waiting time method, the proposed HumLa, and ECo-HumLa, replicating RQs 1.1-1.3 and RQ2.1; 
  - `compute_human_churn()` computes the accumulative code churn for HumLa and Eco-HumLa, replicating RQ2.2;
  - `compute_human_PFs()` computes the human recall~1 defined in Eq.5 and human false alarm defined in Eq.6 of the submitted paper, replicating RQ2.3.





## Usage

### `runRQs.py`

Examples of replicating our RQs are listed below.
````bash
# choose one of the 14 projects contained in the `datasets/` folder.
project_id = 0  # integer \in [0, 13]

# run RQ1.1
RQ1(str_RQ1="RQ1.1", project_id = project_id)

# run RQ1.2
RQ1(str_RQ1="RQ1.2", project_id = project_id)

# run RQ1.3
RQ1(str_RQ1="RQ1.3", project_id = project_id)

# run RQ2.1
RQ2(str_RQ2="RQ2.1", project_id = project_id)

# run RQ2.2
RQ2(str_RQ2="RQ2.2", project_id = project_id)

# run RQ2.3
RQ2(str_RQ2="RQ2.3", project_id = project_id)
````

Notations: 
- See `data_stream/real_data_stream.data_id_2name()` for the project each integer represents.
- One can reduce the values of global variables `N_TEST` and / or `SEEDS` to speed the programme for a general impression.




### `main.run_HumLa()`

We can run HumLa and Eco-HumLa via the function `run_HumLa()` in `main.py`.  Input arguments of `run_HumLa()` are listed as follows.
- `human_dict`: a dictionary -- data structure (see the below examples)
- `project_id`: an integer ranging in [0, 14], each of which represents one project / dataset. See more information from `data_stream/real_data_stream.data_id_2name()`.
- `n_test`: the number of test steps; `n_test=10,000` by default. 
- `seeds`: random seeds and the total number represents the total number of replications; `seeds=range(100)` by default.
- `verbose_int`: an integer to indicate how detailed information would be printed in the console window while running the script; `verbose_int=0` by default. 
  - Larger (smaller) value means that more (less) detailed information would be printed.
  - `verbose_int=-1` means that nothing would be printed to the console.
  - `verbose_int=2` means that the most detailed information can be printed to the console.
- `is_plot`: `True` or `False` to indicate whether 2D plots would be shown during the process; `is_plot=False` by default. 
  - Note `[activity.html](..%2F..%2F..%2F..%2FOneDrive%20-%20%C4%CF%B7%BD%BF%C6%BC%BC%B4%F3%D1%A7%2FCAREER%2FLiyan-career%2FLiyan-CV%40%2Fcareer-Liyan%40%2Fhomepage%2Fgithub-synchronized%40%2Fsunnysong14.github.io%2Factivity.html)is_plot = True` would requre more computational costs.  


Examples of `human_dict` and the way to run the waiting time method, Eco-HumLa, and HumLa at different levels of human noise and effort are as follows.
```bash
# choose a project integer in [0, 13]
project_id = 0 

# the waiting time method
human_dict = {"has_human": False, "human_err": None, "human_eff": None}
run_HumLa(human_dict, project_id)

# ECo-HumLa
human_dict = {"has_human": True, "human_err": 0, "human_eff": "auto_ecohumla2"}
run_HumLa(human_dict, project_id)

# HumLa at 100-%-human effort and 0-human noise
human_dict = {"has_human": True, "human_err": 0, "human_eff": 1}
run_HumLa(human_dict, project_id)

# HumLa at 50%-human effort and 50%-human noise
human_dict = {"has_human": True, "human_err": 0.5, "human_eff": 0.5}
run_HumLa(human_dict, project_id)
```

Note: please be patient that it will take lots of time to complete 100 runs for each dataset throughout the total 10,000 test steps for conducting our systematic experimental studies.
We recommend to try some smaller number for `n_test` and `seeds` to have a quick and general impression of how the programme would proceed.

Examples of conducing such a quick programme are listed as follows.
````bash
project_id = 0 
seeds = range(10)  # 10 times instead of the total 100 times
n_test = 1000  # a shorter data stream

# a quick look at the waiting time method
human_dict = {"has_human": False, "human_err": None, "human_eff": None}
run_HumLa(human_dict, project_id, n_test=n_test, seeds=seeds)

# ECo-HumLa
human_dict = {"has_human": True, "human_err": 0, "human_eff": "auto_ecohumla2"}
run_HumLa(human_dict, project_id, n_test=n_test, seeds=seeds)

# HumLa at 100-%-human effort and 0-human noise
human_dict = {"has_human": True, "human_err": 0, "human_eff": 1}
run_HumLa(human_dict, project_id, n_test=n_test, seeds=seeds)
````



## Other Scrips / Folders
- `data_stream`: this folder has two scripts that are used to produce data streams and preprocess them and one excel that contains the chosen parameter settings based on preliminary experiments. 
- `bagg_ooc`: this method implements Oversampling-based Data Streaming bagging with Confidence (ODaSC) that was adopted whenever JIT-SDP models need to be created / updated. 
- `DenStream`: this folder contains the online clustering method required by ODaSC.
- evaluation_online: this script implements performance metrics such as G-Mean and MCC for the online JIT-SDP.
- `utility`: this script contains utility methods that are used throughout this project.

