# Spatial_Intelligence_Blocks
A block-stacking task built based on robosuite.

## Install
- 首先安装[`robosuite`](https://github.com/ARISE-Initiative/robosuite)，推荐`git clone + pip install -e .`安装，方便定制环境
- 将`si-visual.xml, siRed-visual.xml`复制至`robosuite/robosuite/models/assets/objects/`目录下
- 将`table_arena_si.xml`复制至`robosuite/robosuite/models/assets/arenas/`目录下
- 将环境代码`spatial_intelligence.py`复制至`/lustre/S/tianzikang/rocky/projects/robo_suite/robosuite/robosuite/environments/manipulation/`目录下
- 在`robosuite/robosuite/__init__.py`中添加代码`from robosuite.environments.manipulation.spatial_intelligence import SpatialIntelligence`

## 测试
```bash
./test.sh
```
- 输出如下内容
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
- 生成`task_view`目录，且目录中包含随机生成的积木的三个视角图
- 生成`video`目录，且目录中包含挪动光标搭建`7 * 7 * 7`积木的视频

则表示环境可用

