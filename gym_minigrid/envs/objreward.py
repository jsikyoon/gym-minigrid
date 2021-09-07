import random
from gym_minigrid.minigrid import *
from gym_minigrid.register import register

easy_level_colors = ['green', 'blue', 'purple', 'yellow']
hard_level_colors = ['red', 'green', 'blue', 'purple',
    'yellow', 'grey', 'magenta', 'white', 'orange']

class ObjectRewardEnv(MiniGridEnv):
    """
    It is the 2D version of DmLab30 Object Rewards task.
    The bellow description is from Dmlab git repo.
        This task requires agents to collect objects placed around a room.
        Some objects are from a positive rewarding color, and some are negative.
        After all positive objects are collected, the env restarts.
        Object reward per color is randomised per episode.
    """

    def __init__(
        self,
        size=8,
        numObjs=50,
        level='easy',
        agent_start_pos=(1,1),
        agent_start_dir=0,
        step_penalty=0.0,
    ):
        self.numObjs = numObjs
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.step_penalty = step_penalty

        if level == 'easy':
            self._colormap = easy_level_colors
        elif level == 'hard':
            self._colormap = hard_level_colors
        else:
            raise NotImplementedError

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

        # randomising pos/neg colors
        random.shuffle(self._colormap)
        self._pos_colormap = self._colormap[:len(self._colormap)//2]
        self._neg_colormap = self._colormap[len(self._colormap)//2:]

        # For good objects
        while len(objs) < self.numObjs/2:
            obj = CollectableBall(random.choice(self._pos_colormap), 1)
            self.place_obj(obj)
            objs.append(obj)

        # For bad objects
        while len(objs) < self.numObjs:
            obj = CollectableBall(random.choice(self._neg_colormap), -1)
            self.place_obj(obj)
            objs.append(obj)

        self.mission = "collect the positive color balls that are randomized per each episode."

        self.pos_cnt = 0

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        # Check if we hit a ball
        current_cell = self.grid.get(*self.agent_pos)
        agent_pos = tuple(self.agent_pos)
        if current_cell and current_cell.type == 'ball':
            self.grid.grid[agent_pos[1] * self.grid.width + agent_pos[0]] = None
            reward = current_cell.reward
            current_cell.reward = 0

        reward -= self.step_penalty

        if reward > 0:
            self.pos_cnt += 1

        if self.pos_cnt == self.numObjs/2:
            done = True

        return obs, reward, done, info

class ObjectRewardEasyEnv16x16(ObjectRewardEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, level='easy', **kwargs)

class ObjectRewardEasyRandomEnv16x16(ObjectRewardEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, level='easy', agent_start_pos=None)

class ObjectRewardEasyEnvWithPenalty16x16(ObjectRewardEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, level='easy', step_penalty=0.05, **kwargs)

class ObjectRewardEasyRandomEnvWithPenalty16x16(ObjectRewardEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, level='easy', step_penalty=0.05, agent_start_pos=None)

class ObjectRewardHardEnv16x16(ObjectRewardEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, level='hard', **kwargs)

class ObjectRewardHardRandomEnv16x16(ObjectRewardEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, level='hard', agent_start_pos=None)

class ObjectRewardHardEnvWithPenalty16x16(ObjectRewardEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, level='hard', step_penalty=0.05, **kwargs)

class ObjectRewardHardRandomEnvWithPenalty16x16(ObjectRewardEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, level='hard', step_penalty=0.05, agent_start_pos=None)


register(
    id='MiniGrid-ObjectReward-Easy-16x16-v0',
    entry_point='gym_minigrid.envs:ObjectRewardEasyEnv16x16'
)

register(
    id='MiniGrid-ObjectReward-Easy-Random-16x16-v0',
    entry_point='gym_minigrid.envs:ObjectRewardEasyRandomEnv16x16'
)

register(
    id='MiniGrid-ObjectReward-Easy-Penalty-16x16-v0',
    entry_point='gym_minigrid.envs:ObjectRewardEasyEnvWithPenalty16x16'
)

register(
    id='MiniGrid-ObjectReward-Easy-Random-Penalty-16x16-v0',
    entry_point='gym_minigrid.envs:ObjectRewardEasyRandomEnvWithPenalty16x16'
)

register(
    id='MiniGrid-ObjectReward-Hard-16x16-v0',
    entry_point='gym_minigrid.envs:ObjectRewardHardEnv16x16'
)

register(
    id='MiniGrid-ObjectReward-Hard-Random-16x16-v0',
    entry_point='gym_minigrid.envs:ObjectRewardHardRandomEnv16x16'
)

register(
    id='MiniGrid-ObjectReward-Hard-Penalty-16x16-v0',
    entry_point='gym_minigrid.envs:ObjectRewardHardEnvWithPenalty16x16'
)

register(
    id='MiniGrid-ObjectReward-Hard-Random-Penalty-16x16-v0',
    entry_point='gym_minigrid.envs:ObjectRewardHardRandomEnvWithPenalty16x16'
)

