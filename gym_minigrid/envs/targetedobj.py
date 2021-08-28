from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class TargetedObjectEnv(MiniGridEnv):
    """
    Get a reward when collecting a ball with the same color as the agent.
    The agent's color changes randomly after every ball it collects.
    """

    def __init__(
        self,
        size=8,
        agent_view_size=8,
        numObjs=9,
        agent_start_pos=(1,1),
        agent_start_dir=0,
        step_penalty=0.0,
    ):
        assert numObjs <= len(COLORS)
        self.numObjs = numObjs
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.step_penalty = step_penalty

        super().__init__(
            grid_size=size,
            max_steps=500,
            # Set this to True for maximum speed
            see_through_walls=True,
            agent_view_size=agent_view_size,
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.obj_colors = []

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.agent_color = COLORS['red']

        for obj_idx in range(self.numObjs):
            color = COLOR_NAMES[obj_idx]
            obj = CollectableBall(color, 1)
            self.place_obj(obj)
            self.obj_colors.append(color)

        self.mission = "collect ball that is the same color as the agent"

        self.collected_obj_colors = []

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        reward -= self.step_penalty

        # Check if we hit a ball
        current_cell = self.grid.get(*self.agent_pos)
        if current_cell and current_cell.type == 'ball' and np.array_equal(COLORS[current_cell.color], self.agent_color):
            self.grid.grid[self.agent_pos[1] * self.grid.width + self.agent_pos[0]] = None
            reward = current_cell.reward
            self.collected_obj_colors.append(current_cell.color)
            if len(self.collected_obj_colors) == self.numObjs:
                return obs, reward, True, info

            remaining_colors = [c for c in self.obj_colors if c not in self.collected_obj_colors]
            next_color = np.random.choice(remaining_colors)
            self.agent_color = COLORS[next_color]

        return obs, reward, done, info

class TargetedObjectEnv9x9(TargetedObjectEnv):
    def __init__(self, **kwargs):
        super().__init__(size=9, agent_view_size=5, **kwargs)

class TargetedObjectEnvWithPenalty9x9(TargetedObjectEnv):
    def __init__(self, **kwargs):
        super().__init__(size=9, agent_view_size=5, step_penalty=0.05, **kwargs)


register(
    id='MiniGrid-TargetedObject-9x9-v0',
    entry_point='gym_minigrid.envs:TargetedObjectEnv9x9'
)

register(
    id='MiniGrid-TargetedObject-Penalty-9x9-v0',
    entry_point='gym_minigrid.envs:TargetedObjectEnvWithPenalty9x9'
)


