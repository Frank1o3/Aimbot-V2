import configparser
import ast
import logging
from typing import Dict, Tuple, Union

def load_config(config_file: str) -> Dict[str, Union[str, int, float, bool, Tuple[int, int, int]]]:
    """Load and validate configuration from a file or return defaults."""
    config = {
        "win_name": "Roblox",
        "target_color": (255, 255, 180),
        "fov": 400,
        "sens": 5,
        "offset_head": 0.18,
        "display_fps": True,
        "display_contours": True,
        "lead": 2,
        "min_area": 150,
        "max_targets": 5,
        "process_scale": 0.5,
        "target_switch_interval": 0.5,
        "debug_verbose": True,
        "color_tolerance": 30,
        "smoothing_factor": 0.3,
    }

    if not config_file:
        return config

    parser = configparser.ConfigParser()
    try:
        parser.read(config_file)
        if "Settings" not in parser:
            return config

        settings = parser["Settings"]
        config["win_name"] = settings.get("Name", config["win_name"])
        config["sens"] = _parse_config_int(settings, "Sens", config["sens"], 1, 20)
        config["offset_head"] = _parse_config_float(settings, "Offset_head", config["offset_head"], 0.0, 1.0)
        config["fov"] = _parse_config_int(settings, "Fov", config["fov"], 100, 800)
        try:
            if "Color" in settings:
                color = ast.literal_eval(settings["Color"])
                if isinstance(color, tuple) and len(color) == 3 and all(isinstance(v, int) and 0 <= v <= 255 for v in color):
                    config["target_color"] = color
        except (ValueError, SyntaxError):
            logging.warning(f"Invalid Color value, using default: {config['target_color']}")
        config["lead"] = _parse_config_int(settings, "Lead", config["lead"], 1, 5)
        config["min_area"] = _parse_config_int(settings, "Min_area", config["min_area"], 50, 1000)
        config["max_targets"] = _parse_config_int(settings, "Max_targets", config["max_targets"], 1, 10)
        config["process_scale"] = _parse_config_float(settings, "Process_scale", config["process_scale"], 0.1, 1.0)
        config["target_switch_interval"] = _parse_config_float(settings, "Target_switch_interval", config["target_switch_interval"], 0.1, 2.0)
        config["color_tolerance"] = _parse_config_int(settings, "Color_tolerance", config["color_tolerance"], 10, 60)
        config["smoothing_factor"] = _parse_config_float(settings, "Smoothing_factor", config["smoothing_factor"], 0.1, 0.9)
        config["debug_verbose"] = settings.getboolean("Debug_verbose", config["debug_verbose"])

        if "Debug" in parser:
            debug = parser["Debug"]
            config["display_fps"] = debug.getboolean("FPS", config["display_fps"])
            config["display_contours"] = debug.getboolean("Display", config["display_contours"])
    except Exception as e:
        logging.error(f"Error reading config file: {e}")
        if config["debug_verbose"]:
            print(f"Error reading config file: {e}")
    return config

def _parse_config_int(section: configparser.SectionProxy, key: str, default: int, min_val: int, max_val: int) -> int:
    """Parse and validate an integer from config."""
    try:
        value = int(section.get(key, str(default)))
        return max(min_val, min(value, max_val))
    except ValueError:
        logging.warning(f"Invalid {key} value, using default: {default}")
        return default

def _parse_config_float(section: configparser.SectionProxy, key: str, default: float, min_val: float, max_val: float) -> float:
    """Parse and validate a float from config."""
    try:
        value = float(section.get(key, str(default)))
        return max(min_val, min(value, max_val))
    except ValueError:
        logging.warning(f"Invalid {key} value, using default: {default}")
        return default