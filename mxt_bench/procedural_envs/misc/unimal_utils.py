import brax
from brax.io import file
from google.protobuf import text_format
import numpy as np


def get_config(config_name: str):
    config_path = f"./procedural_envs/components/unimal_configs/{config_name}.txt"
    with file.File(config_path) as f:
        _SYSTEM_CONFIG = f.read()
        f.close()
    return _SYSTEM_CONFIG


# detect end effectors
def get_end_effectors(config_name: str):
    config_path = f"./procedural_envs/components/unimal_configs/{config_name}.txt"
    with file.File(config_path) as f:
        _SYSTEM_CONFIG = f.read()
        f.close()
    config = text_format.Parse(_SYSTEM_CONFIG, brax.Config())
    collides = set([b.name for b in config.bodies])
    parents = set()
    for j in config.joints:
        parents.add(j.parent)
    end_effectors = list(collides - parents)
    return end_effectors


# get all bodies
def get_all_bodies(config_name: str):
    config_path = f"./procedural_envs/components/unimal_configs/{config_name}.txt"
    with file.File(config_path) as f:
        _SYSTEM_CONFIG = f.read()
        f.close()
    config = text_format.Parse(_SYSTEM_CONFIG, brax.Config())
    collides = [b.name for b in config.bodies]
    return collides


def get_agent_names():
    a = np.loadtxt("./procedural_envs/components/unimal_configs/agent_list.csv", delimiter=',', skiprows=1, dtype='str', usecols=[3])
    return a.tolist()
