# Copyright (C) 2023  Andrea Patrizi (AndrePatri, andreapatrizi1b6e6@gmail.com)
# 
# This file is part of OmniCustomGym and distributed under the General Public License version 2 license.
# 
# OmniCustomGym is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# OmniCustomGym is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with OmniCustomGym.  If not, see <http://www.gnu.org/licenses/>.
# 
# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

import omni
from omni.isaac.kit import SimulationApp
import numpy as np

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.tasks import BaseTask
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.utils.prims import define_prim
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.materials import PreviewSurface
from omni.isaac.cloner import GridCloner

from pxr import UsdLux, UsdShade, Sdf

from omni_custom_gym.utils.terrain_utils import *
from omni_custom_gym.utils.terrains import RlTerrains

class TerrainsTest(BaseTask):

    def __init__(self, 
                name) -> None:

        BaseTask.__init__(self, name=name)

        self._device = "cpu"
    
    def set_up_scene(self,
                scene) -> None:
        
        self._stage = get_current_stage()
        distantLight = UsdLux.DistantLight.Define(self._stage, Sdf.Path("/World/DistantLight"))
        distantLight.CreateIntensityAttr(2000)
        
        self.terrains = RlTerrains(self._stage)
        self.terrains.get_obstacles_terrain(
                                    terrain_size = 40.0, 
                                    num_obs = 200, 
                                    max_height = 0.5,
                                    min_size = 0.5,
                                    max_size = 5.0,)

        super().set_up_scene(scene)

        return
    
    def post_reset(self):

        a = 1
        
    def get_observations(self):

        pass

    def calculate_metrics(self) -> None:

        pass

    def is_done(self) -> None:

        pass
    
if __name__ == "__main__":

    world = World(
        stage_units_in_meters=1.0, 
        rendering_dt=1.0/60.0,
        backend="torch", 
        device="cpu",
    )

    terrain_creation_task = TerrainsTest(name="CustomTerrain", 
                                )
                            
    world.add_task(terrain_creation_task)
    world.reset()

    while simulation_app.is_running():
        if world.is_playing():
            if world.current_time_step_index == 0:
                world.reset(soft=True)
            world.step(render=True)
        else:
            world.step(render=True)

    simulation_app.close()