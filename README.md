# OmniCustomGym

Some custom implementations of Tasks and Gyms for [Omniverse Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim.html), photo-realistic GPU accelerated simulatorfrom NVIDIA, based on [Gymnasium](https://gymnasium.farama.org/), a maintained fork of OpenAI’s [Gym ](https://github.com/openai/gym) library (no longer maintained). 
The aim of the package is to provide a standardized interfaces to loading floating-base robots and their configuration from URDF and SRDF and cloning them in Isaac Sim for RL applications. 
If you're also interested in bridging CPU-based controllers with parallel simulations (potentially running on GPU) please have a look at [ControlClusterUtils](https://github.com/AndrePatri/ControlClusterUtils).

The preferred way of using OmniCustomGym package is to employ the provided [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) environment. 

Installation instructions:
- First install Mamba by running ```curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"``` and then ```bash Mambaforge-$(uname)-$(uname -m).sh```.

- Create the mamba environment by running ```create_mamba_env.sh```. This will properly setup a Python 3.7 mamba environment named ```omnicustomgym``` with (almost) all necessary dependencies

- Activate the environment with ```mamba activate omnicustomgym```

- From the root folder install the package in editable mode with ```pip install -e .```.

- Test the Lunar Lander example from StableBaselines3 v2.0 with ```python omnicustomgym/tests/test_lunar_lander_stable_bs3.py```.

- Download [Omniverse Launcer](https://www.nvidia.com/en-us/omniverse/download/), go to the "exchange" tab and install ``` Omniverse Cache``` and  ```Isaac Sim 2022.2.1```  (might take a while). You can then launch it from the Launcher GUI or by navigating to ```${HOME}/.local/share/ov/pkg/isaac_sim-2022.2.1``` and running the ```isaac-sim.sh``` script. When launching IsaacSim for the first time, compilation of ray tracing shaders will take place and may take a while. If the resources of the workstation/pc are limited (e.g. RAM < 16GB), the compilation may abort after a while. You can still manage to compile them by adding sufficient SWAP memory to the system. Before trying to recompile the shaders, remember however to first delete the cache at ```.cache/ov/Kit/*```.

- To be able to run any script with dependencies on Omniverse packages, it's necessary to first source ```${HOME}/.local/share/ov/pkg/isaac_sim-*/setup_conda_env.sh```.

External dependencies to be installed separately: 

- [Omniverse Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim.html), photo-realistic GPU accelerated simulatorfrom NVIDIA.

Other dependencies included in the environment thorough Anaconda which can optionally be installed directly from source for development purposes: 
- [ControlClusterUtils](https://github.com/AndrePatri/ControlClusterUtils): utilities to create CPU-based control cluster to be interfaced with GPU-based simulators using shared memory for minimum latency.
