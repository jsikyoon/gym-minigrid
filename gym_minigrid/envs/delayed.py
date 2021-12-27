from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class DelayedEnv(MiniGridEnv):
    """
    Env where agent gets reward at the end of the episode if it collected the ball earlier in the trajectory.
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
        delay=5, # how many steps to wait until the reward is given
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.delay = delay
        self.reward_steps = None

        super().__init__(
            grid_size=size,
            #max_steps=size*size,
            max_steps=size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        obj = CollectableBall('green', 1)
        self.place_obj(obj)

        self.mission = 'collect the ball for delayed reward'

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        # Check if we hit a ball
        current_cell = self.grid.get(*self.agent_pos)
        agent_pos = tuple(self.agent_pos)
        if current_cell and current_cell.type == 'ball':
            self.grid.grid[agent_pos[1] * self.grid.width + agent_pos[0]] = None
            self.reward_steps = self.delay

        if self.reward_steps is not None:
            if self.reward_steps == 0:
                reward = 1
                #done = True

            self.reward_steps -= 1

        return obs, reward, done, info

class DelayedEnvD5S7x7(DelayedEnv):
    def __init__(self, **kwargs):
        super().__init__(size=7, agent_start_pos=None)

register(
    id='MiniGrid-Delayed-D5-7x7-v0',
    entry_point='gym_minigrid.envs:DelayedEnvD5S7x7'
)

class DelayedEnvD0S7x7(DelayedEnv):
    def __init__(self, **kwargs):
        super().__init__(size=7, agent_start_pos=None, delay=0)

register(
    id='MiniGrid-Delayed-D0-7x7-v0',
    entry_point='gym_minigrid.envs:DelayedEnvD0S7x7'
)

class DelayedEnvD5S9x9(DelayedEnv):
    def __init__(self, **kwargs):
        super().__init__(size=9, agent_start_pos=None)

register(
    id='MiniGrid-Delayed-D5-9x9-v0',
    entry_point='gym_minigrid.envs:DelayedEnvD5S9x9'
)

