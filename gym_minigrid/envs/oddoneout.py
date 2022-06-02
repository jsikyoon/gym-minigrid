import random
import numpy as np
import gym_minigrid.minigrid as minigrid
from gym_minigrid.minigrid import *
from gym_minigrid.register import register


COLORS = ['green', 'blue', 'purple', 'yellow']
SHAPES = ['Ball', 'Box', 'Key']

class OddOneOutEnv(MiniGridEnv):
    """
    This environment to test the reasoning. The agent starts in a large room
    with multiple objects. The objects have 2 properties; color and shape.
    In the objects, only one object has an unique property. For example,
    only one object can be red while others are blue or only one object can be
    square while others are triangle. The agent must understand and only collect
    the object that has the unique property.
    """

    def __init__(
        self,
        seed,
        size=8,
        num_objects=4,
    ):
        self._num_objects = num_objects
        super().__init__(
            seed=seed,
            grid_size=size,
            max_steps=50,
        )

    def _fill_properties(self, objs, unique_property, properties, idx):
        while sum(objs[:,idx]==0) > 0:
            prop = np.random.choice(properties)
            while prop == unique_property:
                prop = np.random.choice(properties)
            num_assigned_objs = np.random.randint(2,sum(objs[:,idx]==0)+1)
            while num_assigned_objs > 0:
                obj_idx = np.random.randint(len(objs))
                if objs[obj_idx, idx] == 0:
                    objs[obj_idx, idx] = prop
                    num_assigned_objs -= 1
            if sum(objs[:,idx]==0) == 1:
                objs[objs[:,idx]==0,idx] = prop
        return objs

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent
        self.place_agent()

        # Give properties to objects
        objs = np.zeros((self._num_objects, 2), dtype=object)
        # Randomly select the unique property
        if np.random.rand() < 0.5:
            unique_property = np.random.choice(COLORS)
            objs[0,0] = unique_property
        else:
            unique_property = np.random.choice(SHAPES)
            objs[0,1] = unique_property
        # Assign colors
        objs = self._fill_properties(objs, unique_property, COLORS, 0)
        objs = self._fill_properties(objs, unique_property, SHAPES, 1)

        # Place objects
        self.place_obj(getattr(minigrid, objs[0][1])(objs[0][0], pos_fruit=True))
        for _obj in objs[1:]:
            self.place_obj(getattr(minigrid, _obj[1])(_obj[0], neg_fruit=True))

        self._target_obj = [_objs[0], _objs[1]]
        self.mission = 'choice the object that has the unique property'

    def step(self, action):
        # to not use other actions except left/right/forward
        if action in [MiniGridEnv.Actions.pickup, MiniGridEnv.Actions.toggle]:
            action = MiniGridEnv.Actions.drop
        obs, reward, done, info = MiniGridEnv.step(self, action)

        if reward != 0:
            done = True

        info.update({'target_obj': self._target_obj})
        return obs, reward, done, info

class OddOneOutS8N4(OddOneOutEnv):
    def __init__(self, num_objects=4, seed=None):
        super().__init__(seed=seed, size=8, num_objects=num_objects)
register(
    id='MiniGrid-OddOneOutS8N4-v0',
    entry_point='gym_minigrid.envs:OddOneOutS8N4',
)

class OddOneOutS8N6(OddOneOutEnv):
    def __init__(self, num_objects=6, seed=None):
        super().__init__(seed=seed, size=8, num_objects=num_objects)
register(
    id='MiniGrid-OddOneOutS8N6-v0',
    entry_point='gym_minigrid.envs:OddOneOutS8N6',
)
