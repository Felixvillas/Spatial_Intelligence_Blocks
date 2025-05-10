# Spatial_Intelligence_Blocks
A block-stacking task built based on robosuite.

## Install
- 首先安装[`robosuite`](https://github.com/ARISE-Initiative/robosuite)，推荐`git clone + pip install -e .`安装，方便定制环境
- 将`si-visual.xml, siBig-visual.xml, siRed-visual.xml`复制至`robosuite/robosuite/models/assets/objects/`目录下
- 将`table_arena_si.xml`复制至`robosuite/robosuite/models/assets/arenas/`目录下
- 将环境代码`spatial_intelligence.py`复制至`robosuite/robosuite/environments/manipulation/`目录下
- 在`robosuite/robosuite/__init__.py`中添加代码`from robosuite.environments.manipulation.spatial_intelligence import SpatialIntelligence`

## 测试
```bash
./test.sh
```
- 输出类似如下内容
```bash
result: True, message: Cube is placed at position [1 0 0].
result: True, message: Cube is placed at position [2 0 0].
result: True, message: Cube is placed at position [3 0 0].
result: True, message: Cube is placed at position [4 0 0].
result: True, message: Cube is placed at position [5 0 0].
result: True, message: Cube is placed at position [6 0 0].
result: True, message: Cursor is moved to position [5 0 0].
...
```
- 生成`task_view`目录，且目录中包含随机生成的积木`cube_xyz_idx`的三个视角图
- 生成`video`目录，且目录中包含挪动光标搭建随机生成的积木`cube_xyz_idx`的视频

则表示环境可用



## 生成connected cubes
已经用`CUDA_VISIBLE_DEVICES=0 python gen_connected_cube_set.py`事先生成了目标积木，搭建对应积木的action序列，以及得出action序列的思考过程，存储在`connected_cube_set_1250_traintest-w_thinks.pkl`中。访问`connected_cube_set_1250_traintest-w_thinks.pkl`的方式如下：

```python
dataset = 1250
traintest = "train"
generated_connected_cube_set = pkl.load(open(os.path.join('$data_dir$', f"connected_cube_set_{dataset}_traintest-w_thinks.pkl")), "rb")[traintest]

for eps in generated_connected_cube_set.keys():
    eps_data = generated_connected_cube_set[eps]
    eps_task = eps_data['task'] # 格式是一个Nx * Ny * Nz的0/1数组，1构成目标积木的外观
    eps_solution = eps_data['solution'] # 格式是一个action list，每个action对应一个step采取的action
    eps_think = eps_data['think'] # 格式是一个str list，每个str对应一个step的reasoning process
```

此外也提供了测试case`connected_cube_set_50_traintest-w_thinks.pkl`供复现，基于sft数据集的每个task对应目标积木的三视图，和搭建积木的过程可视化在`./temp/`目录下。用`CUDA_VISIBLE_DEVICES=0 run_gened_connected_cube_set.py`进行复现


# 用法核心
核心是生成积木的外观，可以试着使用`spatial_intelligence_wrapper.py`的`generate_rubik_by_cube_xyz_idx(self, cube_xyz_idx):`方法，该方法可以根据三位0/1矩阵cube_xyz_idx在环境中拼出来对应积木。对该积木做动作相当于对该0/1矩阵做动作，计算动作之后的0/1矩阵，再次调用该方法可以生成新的外观。