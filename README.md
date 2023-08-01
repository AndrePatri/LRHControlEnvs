# OmniCustomGym

Some custom implementations of Tasks and Gyms for [Omniverse Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim.html), photo-realistic GPU accelerated simulatorfrom NVIDIA, based on [Gymnasium](https://gymnasium.farama.org/), a maintained fork of OpenAI’s [Gym ](https://github.com/openai/gym) library (no longer maintained). 
The aim of the package is to provide a standardized interfaces to loading floating-base robots and their configuration from URDF and SRDF and cloning them in Isaac Sim for RL applications. 
If you're also interested in bridging CPU-based controllers with parallel simulations (potentially running on GPU) please have a look at [ControlClusterUtils](https://github.com/AndPatr/ControlClusterUtils).