# A System for Morphology-Task Generalization via Unified Representation and Behavior Distillation

Accepted to [ICLR2023](https://openreview.net/forum?id=HcUf-QwZeFh) (**notable-top-25%, Spotlight**) [[arxiv]](https://arxiv.org/abs/2211.14296) [[Website]](https://sites.google.com/view/control-graph)

### Citation
If you use this codebase for your research, please cite the paper:

```
@inproceedings{furuta2023asystem,
  title={A System for Morphology-Task Generalization via Unified Representation and Behavior Distillation},
  author={Hiroki Furuta and Yusuke Iwasawa and Yutaka Matsuo and Shixiang Shane Gu},
  booktitle={International Conference on Learning Representations},
  year={2023},
}
```


### Installation
```bash
pip install -r requirements.txt
```


### Behavior Distillation Pipeline
1. Train single-task single-morphology PPO policy on the environment:
```bash
CUDA_VISIBLE_DEVICES=0 python train_ppo_mlp.py --logdir ../results --seed 0 --env ant_reach_4
```

3. Pick trained policy weight, and collect expert `brax.QP`:
```bash
CUDA_VISIBLE_DEVICES=0,1 python generate_behavior_and_qp.py --seed 0 --env ant_reach_4 --task_name ant_reach --params_path ../results/ao_ppo_mlp_single_pro_ant_reach_4_20220707_174507/ppo_mlp_98304000.pkl
```

4. Register `qp_path` (path to saved `brax.QP`) in [dataset_config.py](mxt_bench/procedural_envs/tasks/ant_reach/dataset_config.py).

5. Convert `brax.QP` to morphlogy-task graph representation (e.g. `mtg_v2_base_m`):
```bash
CUDA_VISIBLE_DEVICES=0 python generate_behavior_from_qp.py --seed 0 --env ant_reach_4 --task_name ant_reach --data_name ant_reach_4_mtg_v2_base_m --obs_config2 mtg_v2_base_m
```

6. Register `dataset_path` (path to saved observations) in [dataset_config.py](mxt_bench/procedural_envs/tasks/ant_reach/dataset_config.py) and [task_config.py](mxt_bench/procedural_envs/tasks/task_config.py).

6. Train Transformer policy via multi-task behavior cloning:
```bash
CUDA_VISIBLE_DEVICES=0,1 python train_bc_transformer.py --task_name example --seed 0
# zero-shot evaluation
CUDA_VISIBLE_DEVICES=0,1 python train_bc_transformer_zs.py --task_name example --seed 0
# fine-tuning on multi-task imitation learning
CUDA_VISIBLE_DEVICES=0,1 python train_bc_transformer_fs.py --task_name example --seed 0 --params_path ../results/bc_transformer_zs/policy.pkl
```


### How to Register New Morphology
- Register a blueprint of new morphology in [mxt_bench/procedural_envs/components](mxt_bench/procedural_envs/components) (e.g. [missing ant](mxt_bench/procedural_envs/components/broken_ant.py)).
- If you are interested in custom agents, please follow [unimal.py](mxt_bench/procedural_envs/components/unimal.py).


### How to Register New Task
- See [mxt_bench/procedural_envs/tasks](mxt_bench/procedural_envs/tasks). You need to prepare a dictionary of components (e.g. [ant_reach](mxt_bench/procedural_envs/tasks/ant_reach/ant_reach.py)), and register your task in [register.py](mxt_bench/procedural_envs/tasks/register.py).

```python
ENV_DESCS = dict()

# add environments
for i in range(2, 7, 1):
  ENV_DESCS[f'ant_reach_{i}'] = functools.partial(load_desc, num_legs=i)
  ENV_DESCS[f'ant_reach_hard_{i}'] = functools.partial(load_desc, num_legs=i, r_min=10.5, r_max=11.5)

# missing
for i in range(3, 7, 1):
  for j in range(i):
    ENV_DESCS[f'ant_reach_{i}_b_{j}'] = functools.partial(load_desc, agent='broken_ant', num_legs=i, broken_id=j)
    ENV_DESCS[f'ant_reach_hard_{i}_b_{j}'] = functools.partial(load_desc, agent='broken_ant', num_legs=i, broken_id=j, r_min=10.5, r_max=11.5)
```
- If you would like to avoid immidiate termination after the agents reach to the goal, please set `min_dist=0` in each reward function dict.

### Structure
- [mxt_bench/algo](./mxt_bench/algo/): Algorithms for policy learning (PPO, BC), which supports MLP and Transformer.
- [mxt_bench/models](./mxt_bench/models/): NN architectures.
- [mxt_bench/procedural_envs/components](./mxt_bench/procedural_envs/components): Morphology.
- [mxt_bench/procedural_envs/misc](./mxt_bench/procedural_envs/misc): Utility functions.
- [mxt_bench/procedural_envs/tasks](./mxt_bench/procedural_envs/tasks): Task.
- [mxt_bench/procedural_envs/tasks/observation_config](./mxt_bench/procedural_envs/tasks/observation_config/): Config files for morphlogy-task graph observations.


### Reference
- https://github.com/google/brax
- https://github.com/yobibyte/amorpheus
- https://github.com/agrimgupta92/metamorph
- https://github.com/huangwl18/modular-rl
- https://github.com/WilsonWangTHU/NerveNet
