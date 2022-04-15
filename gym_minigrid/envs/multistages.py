from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class MultiStagesEnv(MiniGridEnv):
    """
    This environment is a memory test. The agent starts in a small room
    where it can see an object. It then moves to other maps, in which,
    it solves one or multiple tasks in limited timesteps.
    At the end, the agent moves to a small room, and it must select one of the given objects,
    one of which is the same as the object in the small room at the beginning.
    The agent has to remember the initial object, and go to the matching
    object.
    """

    def __init__(
        self,
        seed,
        size=13,
        max_steps=50,
        num_stages=3,
    ):
        super().__init__(
            seed=seed,
            grid_size=size,
            max_steps=max_steps,
            agent_view_size=7,
            # Set this to True for maximum speed
            see_through_walls=False,
        )
        self._grid_size = size

    # first stage grid
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        assert height % 2 == 1

        # top wall
        self.grid.set(width//2-1, height // 2 +2, Wall())
        self.grid.set(width//2, height // 2 +2, Wall())
        self.grid.set(width//2+1, height // 2 +2, Wall())
        # botoon wall
        self.grid.set(width//2-1, height // 2 -1, Wall())
        self.grid.set(width//2, height // 2 -1, Wall())
        self.grid.set(width//2+1, height // 2 -1, Wall())
        # left wall
        self.grid.set(width//2-1, height // 2, Wall())
        self.grid.set(width//2-1, height // 2+1, Wall())
        # right wall
        self.grid.set(width//2+1, height // 2, Wall())
        self.grid.set(width//2+1, height // 2+1, Wall())

        # Fix the player's start position and orientation
        self.agent_pos = (width//2, height // 2 + 1)
        self.agent_dir = 3

        # Place objects
        self.start_room_obj = self._rand_elem(['green', 'blue'])
        self.grid.set(width//2, height // 2, Ball(self.start_room_obj))

        self.mission = 'memorize the color and select it at the last step'
        self.stage = 1

    def _gen_second_grid(self, numObjs=50):

        width = height = self._grid_size
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        objs = []

        # Place the agent
        self.place_agent()

        # For good objects
        while len(objs) < int(numObjs/2):
            obj = CollectableBall('green', 0.01)
            self.place_obj(obj)
            objs.append(obj)

        # For bad objects
        while len(objs) < numObjs:
            obj = CollectableBall('blue', -0.01)
            self.place_obj(obj)
            objs.append(obj)

        self.mission = "avoid bad objects and get good objects as many as possible"
        self.stage = 2

    def _gen_third_grid(self):

        width = height = self._grid_size
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        assert height % 2 == 1

        # top wall
        self.grid.set(width//2-1, height // 2 +2, Wall())
        self.grid.set(width//2, height // 2 +2, Wall())
        self.grid.set(width//2+1, height // 2 +2, Wall())
        # botoon wall
        self.grid.set(width//2-1, height // 2 -1, Wall())
        self.grid.set(width//2, height // 2 -1, Wall())
        self.grid.set(width//2+1, height // 2 -1, Wall())
        # left wall
        self.grid.set(width//2-1, height // 2, Wall())
        self.grid.set(width//2-1, height // 2+1, Wall())
        # right wall
        self.grid.set(width//2+1, height // 2, Wall())
        self.grid.set(width//2+1, height // 2+1, Wall())

        # Fix the player's start position and orientation
        self.agent_pos = (width//2, height // 2 + 1)
        self.agent_dir = 3

        # Place objects
        self.last_room_obj = self._rand_elem(['green', 'blue'])
        self.grid.set(width//2, height // 2, Ball(self.last_room_obj))

        self.mission = 'memorize the color and select it at the last step'
        self.stage = 3

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        if self.step_count == 1:
            self._gen_second_grid()
            obs = self.gen_obs()
        # for second stage
        if self.stage == 2:
            # Check if we hit a ball
            current_cell = self.grid.get(*self.agent_pos)
            agent_pos = tuple(self.agent_pos)
            if current_cell and current_cell.type == 'ball':
                self.grid.grid[agent_pos[1] * self.grid.width + agent_pos[0]] = None
                reward = current_cell.reward
                current_cell.reward = 0
        if self.steps_remaining == 1:
            self._gen_third_grid()
            obs = self.gen_obs()
        elif self.steps_remaining == 0:
            if self.start_room_obj == self.last_room_obj:
                if action == 2:
                    reward = 1
                else:
                    reward = -1
            else:
                if action == 2:
                    reward = -1
                else:
                    reward = 1

        return obs, reward, done, info


class MultiStagesS3(MultiStagesEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, num_stages=3)
register(
    id='MiniGrid-MultiStagesS3-v0',
    entry_point='gym_minigrid.envs:MultiStagesS3',
)

