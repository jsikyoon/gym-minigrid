from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class ThreeStagesEnv(MiniGridEnv):
    """
    """

    def __init__(
        self,
        seed,
        grid_size=13,
        max_steps=50,
        level='easy',
    ):
        super().__init__(
            seed=seed,
            grid_size=grid_size,
            max_steps=max_steps,
            agent_view_size=3,
            # Set this to True for maximum speed
            see_through_walls=False,
        )
        self._level = level
        self._grid_size = grid_size

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

    def _gen_second_grid_easy(self):

        width = height = self._grid_size
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        self.mission = 'explore empty space'

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

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        if self.step_count == 1:
            if self._level == 'easy':
                self._gen_second_grid_easy()
            obs = self.gen_obs()
        if self.steps_remaining == 1:
            self._gen_third_grid()
            obs = self.gen_obs()
        elif self.steps_remaining == 0:
            if self.start_room_obj == self.last_room_obj:
                if action == 2:
                    reward = 1
                else:
                    reward = 0
            else:
                if action == 2:
                    reward = 0
                else:
                    reward = 1

        print(self.step_count, self.steps_remaining)

        return obs, reward, done, info


class ThreeStagesEasy(ThreeStagesEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, grid_size=13, max_steps=20, level='easy')
register(
    id='MiniGrid-3StagesEasy-v0',
    entry_point='gym_minigrid.envs:ThreeStagesEasy',
)


