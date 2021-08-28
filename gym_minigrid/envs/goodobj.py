from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class GoodObjectEnv(MiniGridEnv):
    """
    Collect good objects without touching bad objects as many as possible
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        numObjs=50,
        agent_start_pos=(1,1),
        agent_start_dir=0,
        step_penalty=0.0,
        leave_ball_visible=False, # whether or not ball should remain visible after consumed
    ):
        self.numObjs = numObjs
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.step_penalty = step_penalty
        self.leave_ball_visible = leave_ball_visible

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        objs = []

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()


        # For good objects
        while len(objs) < self.numObjs/2:
            obj = CollectableBall('green', 1)
            self.place_obj(obj)
            objs.append(obj)

        # For bad objects
        while len(objs) < self.numObjs:
            obj = CollectableBall('blue', -1)
            self.place_obj(obj)
            objs.append(obj)

        self.mission = "avoid bad objects and get good objects as many as possible"

        self.pos_cnt = 0

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        # Check if we hit a ball
        current_cell = self.grid.get(*self.agent_pos)
        agent_pos = tuple(self.agent_pos)
        if current_cell and current_cell.type == 'ball':
            if not self.leave_ball_visible:
                self.grid.grid[agent_pos[1] * self.grid.width + agent_pos[0]] = None
            reward = current_cell.reward
            current_cell.reward = 0

        reward -= self.step_penalty

        if reward > 0:
            self.pos_cnt += 1

        if self.pos_cnt == self.numObjs/2:
            done = True

        return obs, reward, done, info

class GoodObjectEnv16x16(GoodObjectEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, **kwargs)

class GoodObjectRandomEnv16x16(GoodObjectEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, agent_start_pos=None)

class GoodObjectEnvWithPenalty16x16(GoodObjectEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, step_penalty=0.05, **kwargs)

class GoodObjectRandomEnvWithPenalty16x16(GoodObjectEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, step_penalty=0.05, agent_start_pos=None)

class GoodObjectEnvWithPenalty6x6(GoodObjectEnv):
    def __init__(self, **kwargs):
        super().__init__(size=6, step_penalty=0.05, numObjs=6, **kwargs)

class GoodObjectEnvWithPenalty9x9(GoodObjectEnv):
    def __init__(self, **kwargs):
        super().__init__(size=9, step_penalty=0.05, numObjs=15, **kwargs)

class GoodObjectRandomEnvVisibleBall6x6(GoodObjectEnv):
    def __init__(self, **kwargs):
        super().__init__(size=6, numObjs=6, agent_start_pos=None, leave_ball_visible=True)

class GoodObjectRandomEnvWithPenaltyVisibleBall6x6(GoodObjectEnv):
    def __init__(self, **kwargs):
        super().__init__(size=6, numObjs=6, step_penalty=0.05, agent_start_pos=None, leave_ball_visible=True)

class GoodObjectRandomEnvVisibleBall9x9(GoodObjectEnv):
    def __init__(self, **kwargs):
        super().__init__(size=9, numObjs=15, agent_start_pos=None, leave_ball_visible=True)

class GoodObjectRandomEnvWithPenaltyVisibleBall9x9(GoodObjectEnv):
    def __init__(self, **kwargs):
        super().__init__(size=9, numObjs=15, step_penalty=0.05, agent_start_pos=None, leave_ball_visible=True)

class GoodObjectRandomEnvVisibleBall16x16(GoodObjectEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, agent_start_pos=None, leave_ball_visible=True)

class GoodObjectRandomEnvWithPenaltyVisibleBall16x16(GoodObjectEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, step_penalty=0.05, agent_start_pos=None, leave_ball_visible=True)

register(
    id='MiniGrid-GoodObject-16x16-v0',
    entry_point='gym_minigrid.envs:GoodObjectEnv16x16'
)

register(
    id='MiniGrid-GoodObject-Random-16x16-v0',
    entry_point='gym_minigrid.envs:GoodObjectRandomEnv16x16'
)

register(
    id='MiniGrid-GoodObject-Penalty-16x16-v0',
    entry_point='gym_minigrid.envs:GoodObjectEnvWithPenalty16x16'
)

register(
    id='MiniGrid-GoodObject-Random-Penalty-16x16-v0',
    entry_point='gym_minigrid.envs:GoodObjectRandomEnvWithPenalty16x16'
)

register(
    id='MiniGrid-GoodObject-Penalty-6x6-v0',
    entry_point='gym_minigrid.envs:GoodObjectEnvWithPenalty6x6'
)

register(
    id='MiniGrid-GoodObject-Penalty-9x9-v0',
    entry_point='gym_minigrid.envs:GoodObjectEnvWithPenalty9x9'
)

register(
    id='MiniGrid-GoodObject-Random-VisibleBall-6x6-v0',
    entry_point='gym_minigrid.envs:GoodObjectRandomEnvVisibleBall6x6'
)

register(
    id='MiniGrid-GoodObject-Random-Penalty-VisibleBall-6x6-v0',
    entry_point='gym_minigrid.envs:GoodObjectRandomEnvWithPenaltyVisibleBall6x6'
)

register(
    id='MiniGrid-GoodObject-Random-VisibleBall-9x9-v0',
    entry_point='gym_minigrid.envs:GoodObjectRandomEnvVisibleBall9x9'
)

register(
    id='MiniGrid-GoodObject-Random-Penalty-VisibleBall-9x9-v0',
    entry_point='gym_minigrid.envs:GoodObjectRandomEnvWithPenaltyVisibleBall9x9'
)

register(
    id='MiniGrid-GoodObject-Random-VisibleBall-16x16-v0',
    entry_point='gym_minigrid.envs:GoodObjectRandomEnvVisibleBall16x16'
)

register(
    id='MiniGrid-GoodObject-Random-Penalty-VisibleBall-16x16-v0',
    entry_point='gym_minigrid.envs:GoodObjectRandomEnvWithPenaltyVisibleBall16x16'
)

