# AugMPCEnvs

World interface implementations for the [AugMPC](https://github.com/AndrePatri/AugMPC) package.
Supported environments:
- IsaacSimEnv: main interface for training, built on top of IsaacSim v4.2
- Isaac5xSimEnv: meant to replace IsaaSimEnv, built on top of IsaacSim v5.2 (WIP)
- XMjSimEnv: built on top of [xbot2_mujoco](https://github.com/AndrePatri/xbot2_mujoco), for evaluating sim-sim trasfer on MuJoCo (CPU) using XBot2 middleware, before real-world deployment.
- RtDeploymentEnv: built on top of [xbot2](https://advrhumanoids.github.io/xbot2/master/index.html) and [adarl_ros](git@gitlab.com:crzz/adarl_ros.git), for both real-world deployment. 
