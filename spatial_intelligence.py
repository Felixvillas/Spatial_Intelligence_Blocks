from collections import OrderedDict

import numpy as np
import math

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena, Arena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, UniformFixSampler, SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat
from robosuite.environments.base import MujocoEnv

from robosuite.models.objects import MujocoObject
from copy import deepcopy
from robosuite.utils.mjcf_utils import get_ids
from robosuite.models.world import MujocoWorldBase
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
import robosuite.macros as macros
from robosuite.utils.mjcf_utils import array_to_string, string_to_array, xml_path_completion
from robosuite.models.objects import MujocoXMLObject

class SIVisualObject(MujocoXMLObject):
    """
    Visual fiducial of milk carton (used in PickPlace).

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/si-visual.xml"),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )
        
class SIRedVisualObject(MujocoXMLObject):
    """
    Visual fiducial of milk carton (used in PickPlace).

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/siRed-visual.xml"),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )
        
class SIBigVisualObject(MujocoXMLObject):
    """
    Visual fiducial of milk carton (used in PickPlace).

    Fiducial objects are not involved in collision physics.
    They provide a point of reference to indicate a position.
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/siBig-visual.xml"),
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )
        
class NoRobotTask(MujocoWorldBase):
    def __init__(
        self,
        mujoco_arena,
        mujoco_objects=None,
    ):
        super().__init__()

        # Store references to all models
        self.mujoco_arena = mujoco_arena
        if mujoco_objects is None:
            self.mujoco_objects = []
        else:
            self.mujoco_objects = [mujoco_objects] if isinstance(mujoco_objects, MujocoObject) else mujoco_objects

        # Merge all models
        self.merge_arena(self.mujoco_arena)
        self.merge_objects(self.mujoco_objects)

        self._instances_to_ids = None
        self._geom_ids_to_instances = None
        self._site_ids_to_instances = None
        self._classes_to_ids = None
        self._geom_ids_to_classes = None
        self._site_ids_to_classes = None
        
    def merge_arena(self, mujoco_arena):
        """
        Adds arena model to the MJCF model.

        Args:
            mujoco_arena (Arena): arena to merge into this MJCF model
        """
        self.merge(mujoco_arena)

    def merge_objects(self, mujoco_objects):
        """
        Adds object models to the MJCF model.

        Args:
            mujoco_objects (list of MujocoObject): objects to merge into this MJCF model
        """
        for mujoco_obj in mujoco_objects:
            # Make sure we actually got a MujocoObject
            assert isinstance(mujoco_obj, MujocoObject), "Tried to merge non-MujocoObject! Got type: {}".format(
                type(mujoco_obj)
            )
            # Merge this object
            self.merge_assets(mujoco_obj)
            self.worldbody.append(mujoco_obj.get_obj())

    def generate_id_mappings(self, sim):
        """
        Generates IDs mapping class instances to set of (visual) geom IDs corresponding to that class instance

        Args:
            sim (MjSim): Current active mujoco simulation object
        """
        self._instances_to_ids = {}
        self._geom_ids_to_instances = {}
        self._site_ids_to_instances = {}
        self._classes_to_ids = {}
        self._geom_ids_to_classes = {}
        self._site_ids_to_classes = {}

        models = [model for model in self.mujoco_objects]

        # Parse all mujoco models from robots and objects
        for model in models:
            # Grab model class name and visual IDs
            cls = str(type(model)).split("'")[1].split(".")[-1]
            inst = model.name
            id_groups = [
                get_ids(sim=sim, elements=model.visual_geoms + model.contact_geoms, element_type="geom"),
                get_ids(sim=sim, elements=model.sites, element_type="site"),
            ]
            group_types = ("geom", "site")
            ids_to_instances = (self._geom_ids_to_instances, self._site_ids_to_instances)
            ids_to_classes = (self._geom_ids_to_classes, self._site_ids_to_classes)

            # Add entry to mapping dicts

            # Instances should be unique
            assert inst not in self._instances_to_ids, f"Instance {inst} already registered; should be unique"
            self._instances_to_ids[inst] = {}

            # Classes may not be unique
            if cls not in self._classes_to_ids:
                self._classes_to_ids[cls] = {group_type: [] for group_type in group_types}

            for ids, group_type, ids_to_inst, ids_to_cls in zip(
                id_groups, group_types, ids_to_instances, ids_to_classes
            ):
                # Add geom, site ids
                self._instances_to_ids[inst][group_type] = ids
                self._classes_to_ids[cls][group_type] += ids

                # Add reverse mappings as well
                for idn in ids:
                    assert idn not in ids_to_inst, f"ID {idn} already registered; should be unique"
                    ids_to_inst[idn] = inst
                    ids_to_cls[idn] = cls

    @property
    def geom_ids_to_instances(self):
        """
        Returns:
            dict: Mapping from geom IDs in sim to specific class instance names
        """
        return deepcopy(self._geom_ids_to_instances)

    @property
    def site_ids_to_instances(self):
        """
        Returns:
            dict: Mapping from site IDs in sim to specific class instance names
        """
        return deepcopy(self._site_ids_to_instances)

    @property
    def instances_to_ids(self):
        """
        Returns:
            dict: Mapping from specific class instance names to {geom, site} IDs in sim
        """
        return deepcopy(self._instances_to_ids)

    @property
    def geom_ids_to_classes(self):
        """
        Returns:
            dict: Mapping from geom IDs in sim to specific classes
        """
        return deepcopy(self._geom_ids_to_classes)

    @property
    def site_ids_to_classes(self):
        """
        Returns:
            dict: Mapping from site IDs in sim to specific classes
        """
        return deepcopy(self._site_ids_to_classes)

    @property
    def classes_to_ids(self):
        """
        Returns:
            dict: Mapping from specific classes to {geom, site} IDs in sim
        """
        return deepcopy(self._classes_to_ids)


class SpatialIntelligence(MujocoEnv):
    """
    This class corresponds to the stacking task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        is_gravity=True,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 1.6)) # actuall there is not table in this env, we just use this to set the object position 

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer
        
        # Observations -- Ground truth = object_obs, Image data = camera_obs
        self.use_camera_obs = use_camera_obs

        # Camera / Rendering Settings
        self.has_offscreen_renderer = has_offscreen_renderer
        self.camera_names = (
            list(camera_names) if type(camera_names) is list or type(camera_names) is tuple else [camera_names]
        )
        self.num_cameras = len(self.camera_names)

        self.camera_heights = self._input2list(camera_heights, self.num_cameras)
        self.camera_widths = self._input2list(camera_widths, self.num_cameras)
        self.camera_depths = self._input2list(camera_depths, self.num_cameras)
        self.camera_segmentations = self._input2list(camera_segmentations, self.num_cameras)
        # We need to parse camera segmentations more carefully since it may be a nested list
        seg_is_nested = False
        for i, camera_s in enumerate(self.camera_segmentations):
            if isinstance(camera_s, list) or isinstance(camera_s, tuple):
                seg_is_nested = True
                break
        camera_segs = deepcopy(self.camera_segmentations)
        for i, camera_s in enumerate(self.camera_segmentations):
            if camera_s is not None:
                self.camera_segmentations[i] = self._input2list(camera_s, 1) if seg_is_nested else deepcopy(camera_segs)

        # sanity checks for camera rendering
        if self.use_camera_obs and not self.has_offscreen_renderer:
            raise ValueError("Error: Camera observations require an offscreen renderer!")
        if self.use_camera_obs and self.camera_names is None:
            raise ValueError("Must specify at least one camera name when using camera obs")
        
        self.is_gravity = is_gravity


        super().__init__(
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            renderer=renderer,
            renderer_config=renderer_config,
        )
        
    @staticmethod
    def _input2list(inp, length):
        """
        Helper function that converts an input that is either a single value or a list into a list

        Args:
            inp (None or str or list): Input value to be converted to list
            length (int): Length of list to broadcast input to

        Returns:
            list: input @inp converted into a list of length @length
        """
        # convert to list if necessary
        return list(inp) if type(inp) is list or type(inp) is tuple else [inp for _ in range(length)]
    
    def _create_camera_sensors(self, cam_name, cam_w, cam_h, cam_d, cam_segs, modality="image"):
        """
        Helper function to create sensors for a given camera. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.
        Args:
            cam_name (str): Name of camera to create sensors for
            cam_w (int): Width of camera
            cam_h (int): Height of camera
            cam_d (bool): Whether to create a depth sensor as well
            cam_segs (None or list): Type of segmentation(s) to use, where each entry can be the following:
                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            modality (str): Modality to assign to all sensors
        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given camera
                names (list): array of corresponding observable names
        """
        # Make sure we get correct convention
        convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]

        # Create sensor information
        sensors = []
        names = []

        # Add camera observables to the dict
        rgb_sensor_name = f"{cam_name}_image"
        depth_sensor_name = f"{cam_name}_depth"
        segmentation_sensor_name = f"{cam_name}_segmentation"

        @sensor(modality=modality)
        def camera_rgb(obs_cache):
            img = self.sim.render(
                camera_name=cam_name,
                width=cam_w,
                height=cam_h,
                depth=cam_d,
            )
            if cam_d:
                rgb, depth = img
                obs_cache[depth_sensor_name] = np.expand_dims(depth[::convention], axis=-1)
                return rgb[::convention]
            else:
                return img[::convention]

        sensors.append(camera_rgb)
        names.append(rgb_sensor_name)

        if cam_d:

            @sensor(modality=modality)
            def camera_depth(obs_cache):
                return obs_cache[depth_sensor_name] if depth_sensor_name in obs_cache else np.zeros((cam_h, cam_w, 1))

            sensors.append(camera_depth)
            names.append(depth_sensor_name)

        if cam_segs is not None:
            # Define mapping we'll use for segmentation
            for cam_s in cam_segs:
                seg_sensor, seg_sensor_name = self._create_segementation_sensor(
                    cam_name=cam_name,
                    cam_w=cam_w,
                    cam_h=cam_h,
                    cam_s=cam_s,
                    seg_name_root=segmentation_sensor_name,
                    modality=modality,
                )

                sensors.append(seg_sensor)
                names.append(seg_sensor_name)

        return sensors, names
    
    def _create_segementation_sensor(self, cam_name, cam_w, cam_h, cam_s, seg_name_root, modality="image"):
        """
        Helper function to create sensors for a given camera. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            cam_name (str): Name of camera to create sensors for
            cam_w (int): Width of camera
            cam_h (int): Height of camera
            cam_s (None or list): Type of segmentation to use, should be the following:
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level
            seg_name_root (str): Sensor name root to assign to this sensor

            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                camera_segmentation (function): Generated sensor function for this segmentation sensor
                name (str): Corresponding sensor name
        """
        # Make sure we get correct convention
        convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]

        if cam_s == "instance":
            name2id = {inst: i for i, inst in enumerate(list(self.model.instances_to_ids.keys()))}
            mapping = {idn: name2id[inst] for idn, inst in self.model.geom_ids_to_instances.items()}
        elif cam_s == "class":
            name2id = {cls: i for i, cls in enumerate(list(self.model.classes_to_ids.keys()))}
            mapping = {idn: name2id[cls] for idn, cls in self.model.geom_ids_to_classes.items()}
        else:  # element
            # No additional mapping needed
            mapping = None

        @sensor(modality=modality)
        def camera_segmentation(obs_cache):
            seg = self.sim.render(
                camera_name=cam_name,
                width=cam_w,
                height=cam_h,
                depth=False,
                segmentation=True,
            )
            seg = np.expand_dims(seg[::convention, :, 1], axis=-1)
            # Map raw IDs to grouped IDs if we're using instance or class-level segmentation
            if mapping is not None:
                seg = (
                    np.fromiter(map(lambda x: mapping.get(x, -1), seg.flatten()), dtype=np.int32).reshape(
                        cam_h, cam_w, 1
                    )
                    + 1
                )
            return seg

        name = f"{seg_name_root}_{cam_s}"

        return camera_segmentation, name

    def reward(self, action):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.0 is provided if the red block is stacked on the green block

        Un-normalized components if using reward shaping:

            - Reaching: in [0, 0.25], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube
            - Aligning: in [0, 0.5], encourages aligning one cube over the other
            - Stacking: in {0, 2}, non-zero if cube is stacked on other cube

        The reward is max over the following:

            - Reaching + Grasping
            - Lifting + Aligning
            - Stacking

        The sparse reward only consists of the stacking component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.0 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        r_reach, r_lift, r_stack = self.staged_rewards()
        if self.reward_shaping:
            reward = max(r_reach, r_lift, r_stack)
        else:
            reward = 2.0 if r_stack > 0 else 0.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.0

        return reward

    def staged_rewards(self):
        """
        Helper function to calculate staged rewards based on current physical states.

        Returns:
            3-tuple:

                - (float): reward for reaching and grasping
                - (float): reward for lifting and aligning
                - (float): reward for stacking
        """
        # reaching is successful when the gripper site is close to the center of the cube
        # cubeA_pos = self.sim.data.body_xpos[self.cubeA_body_id]
        # cubeB_pos = self.sim.data.body_xpos[self.cubeB_body_id]
        # gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        # dist = np.linalg.norm(gripper_site_pos - cubeA_pos)
        # r_reach = (1 - np.tanh(10.0 * dist)) * 0.25

        # # grasping reward
        # grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        # if grasping_cubeA:
        #     r_reach += 0.25

        # # lifting is successful when the cube is above the table top by a margin
        # cubeA_height = cubeA_pos[2]
        # table_height = self.table_offset[2]
        # cubeA_lifted = cubeA_height > table_height + 0.04
        # r_lift = 1.0 if cubeA_lifted else 0.0

        # # Aligning is successful when cubeA is right above cubeB
        # if cubeA_lifted:
        #     horiz_dist = np.linalg.norm(np.array(cubeA_pos[:2]) - np.array(cubeB_pos[:2]))
        #     r_lift += 0.5 * (1 - np.tanh(horiz_dist))

        # # stacking is successful when the block is lifted and the gripper is not holding the object
        # r_stack = 0
        # cubeA_touching_cubeB = self.check_contact(self.cubeA, self.cubeB)
        # if not grasping_cubeA and r_lift > 0 and cubeA_touching_cubeB:
        #     r_stack = 2.0

        # return r_reach, r_lift, r_stack
        return 0, 0, 0

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # load model for table top workspace
        # mujoco_arena = TableArena(
        #     table_full_size=self.table_full_size,
        #     table_friction=self.table_friction,
        #     table_offset=self.table_offset,
        # )
        mujoco_arena = Arena(
            xml_path_completion(xml_path="arenas/table_arena_si.xml")
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        
        self.cubes, self.color_cube = [], None
        rubik_x_size, rubik_y_size, rubik_z_size = 7, 7, 7 # a rubik's cube
        assert (rubik_x_size == rubik_y_size and rubik_y_size == rubik_z_size and rubik_z_size % 2 == 1)
        # generate cubes' positions from the rubik's low-level to high-level
        # each cube's size is (x_size, y_size, z_size) = (0.04 0.04 0.04)
        # the center cube's position is (0, 0, 0.0)
        rubik_position = {
            f"{x}_{y}_{z}": None for x in range(rubik_x_size) for y in range(rubik_y_size) for z in range(rubik_z_size)
        }
        cube_x_size, cube_y_size, cube_z_size = 0.04, 0.04, 0.04
        base_offset = [0, 0, -2] # make the cube in the initial position invisible to the camera
        center_x_idx, center_y_idx, center_z_idx = rubik_x_size // 2, rubik_y_size // 2, rubik_z_size // 2
        rubik_position[f"{center_x_idx}_{center_y_idx}_{center_z_idx}"] = np.array([0, 0, 0])
        for z in range(rubik_z_size):
            for y in range(rubik_y_size):
                for x in range(rubik_x_size):
                    x_pos = (x - center_x_idx) * cube_x_size + base_offset[0]
                    y_pos = (y - center_y_idx) * cube_y_size + base_offset[1]
                    z_pos = (z - center_z_idx) * cube_z_size + base_offset[2]
                    rubik_position[f"{x}_{y}_{z}"] = np.array([x_pos, y_pos, z_pos])
        rubik_position["red_cube"] = rubik_position["0_0_0"] + np.array([0, 0, -base_offset[2]])
        
        cube_name_prefix = "virtual_cube"
        
        if cube_name_prefix == "cube":
            tex_attrib = {
                "type": "cube",
            }
            mat_attrib = {
                "texrepeat": "1 1",
                "specular": "0.4",
                "shininess": "0.1",
            }
            greenwood = CustomMaterial(
                texture="WoodGreen",
                tex_name="greenwood",
                mat_name="greenwood_mat",
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib,
            )
            
        for z in range(rubik_z_size):
            for y in range(rubik_y_size):
                for x in range(rubik_x_size):
                    self.cubes.append(
                        SIVisualObject(name=f"{cube_name_prefix}_{x}_{y}_{z}")
                    )
                    self.placement_initializer.append_sampler(UniformFixSampler(
                        name=f"{cube_name_prefix}_{x}_{y}_{z}",
                        mujoco_objects=self.cubes[-1],
                        x_range=[rubik_position[f"{x}_{y}_{z}"][0], rubik_position[f"{x}_{y}_{z}"][0]],
                        y_range=[rubik_position[f"{x}_{y}_{z}"][1], rubik_position[f"{x}_{y}_{z}"][1]],
                        rotation=math.pi / 2,
                        rotation_axis='z',
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=self.table_offset,
                        z_offset=rubik_position[f"{x}_{y}_{z}"][2]
                    ))
            
        # create the red cube
        self.color_cube = SIRedVisualObject(name="virtual_cube_red_cube")
        self.placement_initializer.append_sampler(UniformFixSampler(
            name="virtual_cube_red_cube",
            mujoco_objects=self.color_cube,
            x_range=[rubik_position["red_cube"][0], rubik_position["red_cube"][0]],
            y_range=[rubik_position["red_cube"][1], rubik_position["red_cube"][1]],
            rotation=math.pi / 2,
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=rubik_position["red_cube"][2]
        ))
        
        self.initial_rubik_position = {key: value + self.table_offset for key, value in rubik_position.items() if key != "red_cube"}
        self.final_rubik_position = {key: value + np.array([0, 0, -base_offset[2]]) for key, value in self.initial_rubik_position.items()}
        
        # create the virtual cube for indicating the rubik's boundary
        self.virtual_big_cube = SIBigVisualObject(name="virtual_big_cube")
        center_cube_idx_str = f"{center_x_idx}_{center_y_idx}_{center_z_idx}"
        virtual_big_cube_pos = self.final_rubik_position[center_cube_idx_str] - self.table_offset
        self.placement_initializer.append_sampler(UniformFixSampler(
            name="virtual_big_cube",
            mujoco_objects=self.virtual_big_cube,
            x_range=[virtual_big_cube_pos[0], virtual_big_cube_pos[0]],
            y_range=[virtual_big_cube_pos[1], virtual_big_cube_pos[1]],
            rotation=0,
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=virtual_big_cube_pos[2]
        ))
        
        # red cube's initial xyz index
        self.rubik_x_size = rubik_x_size
        self.rubik_y_size = rubik_y_size
        self.rubik_z_size = rubik_z_size
        self._reset_vars()
        
        # task includes arena, robot, and objects of interest
        self.model = NoRobotTask(
            mujoco_arena=mujoco_arena,
            mujoco_objects=self.cubes + [self.color_cube, self.virtual_big_cube],
        )
        
    def _reset_vars(self):
        self.rubik_red_cube_xyz_idx = None # red cube's xyz index
        self.rubik_xyz_idx_exists = np.zeros((self.rubik_x_size, self.rubik_y_size, self.rubik_z_size), dtype=bool)

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        # self.cubeA_body_id = self.sim.model.body_name2id(self.cubeA.root_body)
        # self.cubeB_body_id = self.sim.model.body_name2id(self.cubeB.root_body)
        self.obj_body_id = {}
        self.obj_geom_id = {}
        for obj in self.cubes + [self.color_cube, self.virtual_big_cube]:
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)
            self.obj_geom_id[obj.name] = [self.sim.model.geom_name2id(g) for g in obj.contact_geoms]

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        
        # reset the self.rubik_xyz_idx_exists and self.rubik_red_cube_xyz_idx
        self._reset_vars()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                if "virtual" in obj.name.lower():
                    """
                    ATTENTION: JimuVisualObject can use this branch; JimuObject cannot use this branch, 
                    will make self.sim.data.body_xpos and self.sim.data.body_xquat
                    """
                    self.sim.model.body_pos[self.obj_body_id[obj.name]] = obj_pos
                    self.sim.model.body_quat[self.obj_body_id[obj.name]] = obj_quat
                else:
                    """
                    ATTENTION: JimuObject can use this branch; JimuVisualObject cannot use this branch, 
                    will make IndexError: list index out of range
                    """
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()
        # Loop through all robots and grab their observables, adding it to the proprioception modality

        # Loop through cameras and update the observations if using camera obs
        if self.use_camera_obs:
            # Create sensor information
            sensors = []
            names = []
            for (cam_name, cam_w, cam_h, cam_d, cam_segs) in zip(
                self.camera_names,
                self.camera_widths,
                self.camera_heights,
                self.camera_depths,
                self.camera_segmentations,
            ):

                # Add cameras associated to our arrays
                cam_sensors, cam_sensor_names = self._create_camera_sensors(
                    cam_name, cam_w=cam_w, cam_h=cam_h, cam_d=cam_d, cam_segs=cam_segs, modality="image"
                )
                sensors += cam_sensors
                names += cam_sensor_names

            # If any the camera segmentations are not None, then we shrink all the sites as a hacky way to
            # prevent them from being rendered in the segmentation mask
            if not all(seg is None for seg in self.camera_segmentations):
                self.sim.model.site_size[:, :] = 1.0e-8

            # Create observables for these cameras
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        # # low-level object information
        # if self.use_object_obs:
        #     # Get robot prefix and define observables modality
        #     pf = self.robots[0].robot_model.naming_prefix
        #     modality = "object"

        #     # position and rotation of the first cube
        #     @sensor(modality=modality)
        #     def cubeA_pos(obs_cache):
        #         return np.array(self.sim.data.body_xpos[self.cubeA_body_id])

        #     @sensor(modality=modality)
        #     def cubeA_quat(obs_cache):
        #         return convert_quat(np.array(self.sim.data.body_xquat[self.cubeA_body_id]), to="xyzw")

        #     @sensor(modality=modality)
        #     def cubeB_pos(obs_cache):
        #         return np.array(self.sim.data.body_xpos[self.cubeB_body_id])

        #     @sensor(modality=modality)
        #     def cubeB_quat(obs_cache):
        #         return convert_quat(np.array(self.sim.data.body_xquat[self.cubeB_body_id]), to="xyzw")

        #     @sensor(modality=modality)
        #     def gripper_to_cubeA(obs_cache):
        #         return (
        #             obs_cache["cubeA_pos"] - obs_cache[f"{pf}eef_pos"]
        #             if "cubeA_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
        #             else np.zeros(3)
        #         )

        #     @sensor(modality=modality)
        #     def gripper_to_cubeB(obs_cache):
        #         return (
        #             obs_cache["cubeB_pos"] - obs_cache[f"{pf}eef_pos"]
        #             if "cubeB_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
        #             else np.zeros(3)
        #         )

        #     @sensor(modality=modality)
        #     def cubeA_to_cubeB(obs_cache):
        #         return (
        #             obs_cache["cubeB_pos"] - obs_cache["cubeA_pos"]
        #             if "cubeA_pos" in obs_cache and "cubeB_pos" in obs_cache
        #             else np.zeros(3)
        #         )

        #     sensors = [cubeA_pos, cubeA_quat, cubeB_pos, cubeB_quat, gripper_to_cubeA, gripper_to_cubeB, cubeA_to_cubeB]
        #     names = [s.__name__ for s in sensors]

        #     # Create observables
        #     for name, s in zip(names, sensors):
        #         observables[name] = Observable(
        #             name=name,
        #             sensor=s,
        #             sampling_rate=self.control_freq,
        #         )

        return observables

    def _check_success(self):
        """
        Check if blocks are stacked correctly.

        Returns:
            bool: True if blocks are correctly stacked
        """
        _, _, r_stack = self.staged_rewards()
        return r_stack > 0

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # # Color the gripper visualization site according to its distance to the cube
        # if vis_settings["grippers"]:
        #     # self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cubeA)
        #     self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.color_cube)
            
            
    def set_cube_joint(self, cube_name, cube_pos, cube_quat=None):
        """
        Set the joint position of the robot.

        Args:
            name (str): Name of the joint to set
            joint_pos (float): Value to set the joint to
        """
        assert cube_quat is None, "cube_quat should be None"
        if "virtual" in cube_name.lower():
            self.sim.model.body_pos[self.obj_body_id[cube_name]] = cube_pos
            if cube_quat is not None:
                self.sim.model.body_quat[self.obj_body_id[cube_name]] = cube_quat
        else:
            if cube_quat is None:
                cube_quat = self.sim.data.body_xquat[self.obj_body_id[cube_name]]
            self.sim.data.set_joint_qpos(self.cubes[0].joints[0], np.concatenate([np.array(cube_pos), np.array(cube_quat)]))
            
        self.sim.forward() # this is necessary to update the simulation state after the object is set

    
    def place_cube(self, direction, target_cube_xyz_idx):
        placed_cube_xyz_idx = self.rubik_red_cube_xyz_idx.copy()
        if direction == "up":
            placed_cube_xyz_idx[2] += 1
        elif direction == "down":
            placed_cube_xyz_idx[2] -= 1
        elif direction == "left":
            placed_cube_xyz_idx[1] -= 1
        elif direction == "right":
            placed_cube_xyz_idx[1] += 1
        elif direction == "forward":
            placed_cube_xyz_idx[0] += 1
        elif direction == "backward":
            placed_cube_xyz_idx[0] -= 1
        else:
            return {
                "success": None,
                "message": f"Invalid direction: {direction}. Please use one of the following directions: up, down, left, right, forward, backward."
            }
        
        if placed_cube_xyz_idx[0] >= 0 and placed_cube_xyz_idx[0] < self.rubik_x_size and \
            placed_cube_xyz_idx[1] >= 0 and placed_cube_xyz_idx[1] < self.rubik_y_size and \
            placed_cube_xyz_idx[2] >= 0 and placed_cube_xyz_idx[2] < self.rubik_z_size:
                
            if not target_cube_xyz_idx[placed_cube_xyz_idx[0], placed_cube_xyz_idx[1], placed_cube_xyz_idx[2]]:
                return {
                    "success": False,
                    "message": f"The target block has no Cube at {direction} direction."
                }
                
            if self.is_gravity:
                if placed_cube_xyz_idx[2] - 1 >= 0 and \
                    not self.rubik_xyz_idx_exists[placed_cube_xyz_idx[0], placed_cube_xyz_idx[1], placed_cube_xyz_idx[2] - 1]:
                    return {
                        "success": False,
                        "message": f"Cube at {direction} direction cannot be placed because there is no cube below it."
                    }
            
            if self.rubik_xyz_idx_exists[placed_cube_xyz_idx[0], placed_cube_xyz_idx[1], placed_cube_xyz_idx[2]]:
                return {
                    "success": False,
                    "message": f"Cube at {direction} direction already exists."
                }
            # move the red cube to the new position
            self.set_cube_joint(
                cube_name="virtual_cube_red_cube",
                cube_pos=self.final_rubik_position[f"{placed_cube_xyz_idx[0]}_{placed_cube_xyz_idx[1]}_{placed_cube_xyz_idx[2]}"],
            )
            self.rubik_xyz_idx_exists[placed_cube_xyz_idx[0], placed_cube_xyz_idx[1], placed_cube_xyz_idx[2]] = True
            # move the cube in the red cube's last position from the initial position to the final position
            self.set_cube_joint(
                cube_name=f"virtual_cube_{self.rubik_red_cube_xyz_idx[0]}_{self.rubik_red_cube_xyz_idx[1]}_{self.rubik_red_cube_xyz_idx[2]}",
                cube_pos=self.final_rubik_position[f"{self.rubik_red_cube_xyz_idx[0]}_{self.rubik_red_cube_xyz_idx[1]}_{self.rubik_red_cube_xyz_idx[2]}"],
            )
            self.rubik_red_cube_xyz_idx = placed_cube_xyz_idx.copy()
            return {
                "success": True,
                "message": f"Cube is placed at {direction} direction."
            }
        return {
            "success": False,
            "message": f"The {direction} direction of cube to be placed is out of the operate boundary."
        }
        
    def move_red_cube(self, direction):
        rubik_red_cube_xyz_idx = self.rubik_red_cube_xyz_idx.copy()
        if direction == "up":
            rubik_red_cube_xyz_idx[2] += 1
        elif direction == "down":
            rubik_red_cube_xyz_idx[2] -= 1
        elif direction == "left":
            rubik_red_cube_xyz_idx[1] -= 1
        elif direction == "right":
            rubik_red_cube_xyz_idx[1] += 1
        elif direction == "forward":
            rubik_red_cube_xyz_idx[0] += 1
        elif direction == "backward":
            rubik_red_cube_xyz_idx[0] -= 1
        else:
            return {
                "success": None,
                "message": f"Invalid direction: {direction}. Please use one of the following directions: up, down, left, right, forward, backward."
            }
        
        if rubik_red_cube_xyz_idx[0] >= 0 and rubik_red_cube_xyz_idx[0] < self.rubik_x_size and \
            rubik_red_cube_xyz_idx[1] >= 0 and rubik_red_cube_xyz_idx[1] < self.rubik_y_size and \
            rubik_red_cube_xyz_idx[2] >= 0 and rubik_red_cube_xyz_idx[2] < self.rubik_z_size:
            
            if self.rubik_xyz_idx_exists[rubik_red_cube_xyz_idx[0], rubik_red_cube_xyz_idx[1], rubik_red_cube_xyz_idx[2]]:
                # move the cube to the initial position
                self.set_cube_joint(
                    cube_name=f"virtual_cube_{rubik_red_cube_xyz_idx[0]}_{rubik_red_cube_xyz_idx[1]}_{rubik_red_cube_xyz_idx[2]}",
                    cube_pos=self.initial_rubik_position[f"{rubik_red_cube_xyz_idx[0]}_{rubik_red_cube_xyz_idx[1]}_{rubik_red_cube_xyz_idx[2]}"],
                )
                # then move the red cube to the new position
                self.set_cube_joint(
                    cube_name="virtual_cube_red_cube",
                    cube_pos=self.final_rubik_position[f"{rubik_red_cube_xyz_idx[0]}_{rubik_red_cube_xyz_idx[1]}_{rubik_red_cube_xyz_idx[2]}"],
                )
                # then move the cube in the red cube's last position from the initial position to the final position
                self.set_cube_joint(
                    cube_name=f"virtual_cube_{self.rubik_red_cube_xyz_idx[0]}_{self.rubik_red_cube_xyz_idx[1]}_{self.rubik_red_cube_xyz_idx[2]}",
                    cube_pos=self.final_rubik_position[f"{self.rubik_red_cube_xyz_idx[0]}_{self.rubik_red_cube_xyz_idx[1]}_{self.rubik_red_cube_xyz_idx[2]}"],
                ) 
                self.rubik_red_cube_xyz_idx = rubik_red_cube_xyz_idx.copy()
                return {
                    "success": True,
                    "message": f"Cursor is moved to {direction} direction."
                }
            else:
                return {
                    "success": False,
                    "message": f"Cursor cannot be moved to {direction} direction because there is no cube at {direction}."
                }
                
        return {
            "success": False,
            "message": f"The {direction} direction Cursor try to move to is out of the operate boundary."
        }
                        