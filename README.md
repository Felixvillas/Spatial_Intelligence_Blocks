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



## generated connected cubes
已经用`CUDA_VISIBLE_DEVICES=0 python gen_connected_cube_set.py`事先生成了目标积木和搭建对应积木的action序列，存储在`connected_cube_set_100.pkl`和`connected_cube_set_1000.pkl`中，分别表示100个eps和1000个eps

可以用`run_gened_connected_cube_set.py`复现事先生成的积木，但是需要将196-199行的注释取消，然后将200注释上，然后将38行的"test"改为"train"