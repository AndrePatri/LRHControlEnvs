import os

class PathsGetter:

    def __init__(self):
        
        self.PACKAGE_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
        
        self.OMNI_APPKITS_ROOT = os.path.join(self.PACKAGE_ROOT_DIR, 
                                    'cfg',
                                    'omni_kits')
        self.OMNIRGYM_HEADLESS_KIT = os.path.join(self.OMNI_APPKITS_ROOT, 
                                    'omni.isaac.sim.python.lrhcontrolenvs.headless.kit')
        self.OMNIRGYM_KIT = os.path.join(self.OMNI_APPKITS_ROOT, 
                                    'omni.isaac.sim.python.lrhcontrolenvs.kit')