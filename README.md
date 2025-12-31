# context-f110

This repo contains a environment for context-aware head-to-head F1Tenth/Roboracer. I adatped this environment from the open source implementation of [this paper](https://ieeexplore.ieee.org/document/10182327). I developed this environment for part of my Master's thesis and [this resulting work](https://arxiv.org/abs/2510.11501), both supervised by Prof Ivana Dusparic at Trinity College Dublin.

Currently, the repo contains an implementation of the [f110_gym](https://par.nsf.gov/biblio/10221872) environment, implementations of MFRL and MBRL algorithms, implementations of context-defined rules-based racing algorithms, as well as training and evaluation scripts for the development of RL and control policies in a context-aware environment.

![](Data/overtake.gif)

## Setup
To setup the environment, I would advise you first to install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

Once this is complete, please create a new conda environment using the ```environment.yml``` file I have provided. This can be done by using the following command.

```bash
conda env create -f environment.yml
```

The conda environment is called ```f110-gym```. The environment can be activated by simply using the following command.

```bash
conda activate f110-gym
```

## Training & Evaluation
There are two important files for running experiments. These are ```TrainAgents.py``` and ```ContextAwareF110/TestSimulation.py```. These scripts execute training and evaluation, respectively. Both scripts are configured using the ```.yaml``` files in the ```config/``` folder. To execute the training or evaluation of a given configuration, run the command below with an appropriate config file (e.x. ```dev```). 

```python
## To run training
python TrainAgents.py -r [CONFIG]

## To run evaluation
python ContextAwareF110/TestSimulation.py -r [CONFIG]
```

The configuration files specify the trainable and adversarial agent types, the number of training episodes, the number of evaluation episodes, etc. The range of contexts used during training and evaluation is also defined in these scripts using the ```context_info``` tag. The ```context_info``` parameter takes a four-element array in which the first two numbers indicate the magnitude of the range of speed and steering contexts, respectively, and the final two represent the ranges used during evaluation.  

## Citation

If you find this work useful, please consider citing:
```
@misc{moustafa2025contextawaremodelbasedreinforcementlearning,
      title={Context-Aware Model-Based Reinforcement Learning for Autonomous Racing}, 
      author={Emran Yasser Moustafa and Ivana Dusparic},
      year={2025},
      eprint={2510.11501},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.11501}, 
}
```


