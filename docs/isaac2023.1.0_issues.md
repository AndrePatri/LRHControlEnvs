### Some bugs of Isaac2023.1.0 which can be easily fixed

#### 1.0 Nucleus blocking function makes startup super slow
Easy temporary fix: modify /home/username/.local/share/ov/pkg/isaac_sim-2023.1.0/exts/omni.isaac.core/omni/isaac/core/utils/nucleus.py .

Change lines 178 to 198 which is the check server function to below:
```python
def check_server(server: str, path: str) -> bool:
    """Check a specific server for a path

    Args:
        server (str): Name of Nucleus server
        path (str): Path to search

    Returns:
        bool: True if folder is found
    """
    carb.log_info("Checking path: {}{}".format(server, path))
    # Increase hang detection timeout
    if "localhost" not in server:
        omni.client.set_hang_detection_time_ms(10000)
        result, _ = omni.client.stat("{}{}".format(server, path))
        if result == Result.OK:
            carb.log_info("Success: {}{}".format(server, path))
            return True
    carb.log_info("Failure: {}{} not accessible".format(server, path))
    return False
```

#### 2.0 Grid Cloner bug
See `docs/grid_cloner_bugfix.py` for more details
