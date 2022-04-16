from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class IMazeEnv(MiniGridEnv):
    """
    This environment is a memory test. The agent starts in a small corridor
    where it sees an object. It then has to go through a narrow hallway
    which ends in a split. At each end of the split there is an object,
    one of which is the same as the object in the starting room. The
    agent has to remember the initial object, and go to the matching
    object at split.
    """

    def __init__(
        self,
        seed,
        size=8,
        max_steps=100
    ):
        super().__init__(
            seed=seed,
            grid_size=size,
            #max_steps=10*size,
            max_steps=max_steps,
            agent_view_size=3,
            # Set this to True for maximum speed
            see_through_walls=False,
        )

    def _gen_grid(self, width, height):
        self._width = width
        self._height = height
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        assert height % 2 == 1
        upper_room_wall = height // 2 - 2
        lower_room_wall = height // 2 + 2
        hallway_end = width - 3

        for i in range(1, 3):
            self.grid.set(i, upper_room_wall, Wall())
            self.grid.set(i, lower_room_wall, Wall())
        self.grid.set(2, upper_room_wall + 1, Wall())
        self.grid.set(2, lower_room_wall - 1, Wall())

        # Horizontal hallway
        for i in range(3, hallway_end):
            self.grid.set(i, upper_room_wall + 1, Wall())
            self.grid.set(i, lower_room_wall - 1, Wall())

        # Vertical hallway
        for j in range(0, height):
            if j != height // 2:
                self.grid.set(hallway_end, j, Wall())
            self.grid.set(hallway_end + 2, j, Wall())

        # Fix the player's start position and orientation
        self.agent_pos = (1, height // 2 + 1)
        self.agent_dir = 3

        # Place objects
        start_room_obj = self._rand_elem(['green', 'blue'])
        self.grid.set(1, height // 2 - 1, Ball(start_room_obj))

        other_objs = self._rand_elem([['green', 'blue'], ['green', 'blue']])
        #pos0 = (hallway_end + 1, height // 2 - 4)
        pos0 = (hallway_end + 1, height // 2 - 2)
        #pos1 = (hallway_end + 1, height // 2 + 4)
        pos1 = (hallway_end + 1, height // 2 + 2)
        self.grid.set(*pos0, Ball(other_objs[0]))
        self.grid.set(*pos1, Ball(other_objs[1]))

        # Choose the target objects
        if start_room_obj == other_objs[0]:
            self.success_pos = (pos0[0], pos0[1] + 1)
            self.failure_pos = (pos1[0], pos1[1] - 1)
        else:
            self.success_pos = (pos1[0], pos1[1] - 1)
            self.failure_pos = (pos0[0], pos0[1] + 1)

        self.mission = 'go to the matching object at the end of the hallway'

        self.additional_reward_set = [0.01] * (width-2)

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        if self.agent_pos[1] == self._height//2:
            reward = self.additional_reward_set[self.agent_pos[0]-1]
            self.additional_reward_set[self.agent_pos[0]-1] = 0

        if tuple(self.agent_pos) == self.success_pos:
            #reward = self._reward()
            reward = 10
            done = True
        if tuple(self.agent_pos) == self.failure_pos:
            reward = 0.2
            done = True

        return obs, reward, done, info


class IMazeS5(IMazeEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=5, max_steps=7)
register(
    id='MiniGrid-IMazeS5-v0',
    entry_point='gym_minigrid.envs:IMazeS5',
)


class IMazeS9(IMazeEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=9, max_steps=15)
register(
    id='MiniGrid-IMazeS9-v0',
    entry_point='gym_minigrid.envs:IMazeS9',
)


class IMazeS13(IMazeEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=13, max_steps=200)
register(
    id='MiniGrid-IMazeS13-v0',
    entry_point='gym_minigrid.envs:IMazeS13',
)


class IMazeS21(IMazeEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=21, max_steps=300)
register(
    id='MiniGrid-IMazeS21-v0',
    entry_point='gym_minigrid.envs:IMazeS21',
)


class IMazeS31(IMazeEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=31, max_steps=400)
register(
    id='MiniGrid-IMazeS31-v0',
    entry_point='gym_minigrid.envs:IMazeS31',
)

class IMazeS41(IMazeEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=41, max_steps=50)
register(
    id='MiniGrid-IMazeS41-v0',
    entry_point='gym_minigrid.envs:IMazeS41',
)

class IMazeS51(IMazeEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=51, max_steps=60)
register(
    id='MiniGrid-IMazeS51-v0',
    entry_point='gym_minigrid.envs:IMazeS51',
)

class IMazeS61(IMazeEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=61, max_steps=70)
register(
    id='MiniGrid-IMazeS61-v0',
    entry_point='gym_minigrid.envs:IMazeS61',
)

class IMazeS71(IMazeEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=71, max_steps=80)
register(
    id='MiniGrid-IMazeS71-v0',
    entry_point='gym_minigrid.envs:IMazeS71',
)

class IMazeS81(IMazeEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=81, max_steps=90)
register(
    id='MiniGrid-IMazeS81-v0',
    entry_point='gym_minigrid.envs:IMazeS81',
)

class IMazeS91(IMazeEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=91, max_steps=100)
register(
    id='MiniGrid-IMazeS91-v0',
    entry_point='gym_minigrid.envs:IMazeS91',
)


