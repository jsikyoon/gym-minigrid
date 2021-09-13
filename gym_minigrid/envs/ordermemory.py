import random, copy
from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class OrderMemoryEnv(MiniGridEnv):
    """
    Collect the objects in hidden order.
    If the agent collects the object out of the hidden order,
      the map is reset and it need to collect from the first.
    But, the reward for each object is given once only.
    If the agent collects every objects in the order,
      the map and rewards for objects are reset.
    The agent collects the objects in the order as many as possible in max length.
    """

    def __init__(
        self,
        numObjs=3,
        step_penalty=0.0,
    ):
        if numObjs == 3: #N3
            size = 5
            max_steps = 100
            ball_colors = ['green', 'blue', 'yellow']
            poses = [(1,3), (2,2), (3,3)]
            agent_start_pos = (2,3)
            agent_start_dir = 3
        elif numObjs == 4: #N4
            size = 5
            max_steps = 100
            ball_colors = ['green', 'blue', 'yellow', 'purple']
            poses = [(2,1), (3,2), (2,3), (1,2)]
            agent_start_pos = (2,2)
            agent_start_dir = 3
        else:
            raise NotImplementedError

        self.numObjs = numObjs
        self.size = size
        self.max_steps = max_steps
        self.ball_colors = ball_colors
        self.poses = poses
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.step_penalty = step_penalty

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        # Place objects
        random.shuffle(self.ball_colors)
        for _pos, color in zip(self.poses, self.ball_colors):
            self.grid.set(*_pos, CollectableBall(color, 0))

        # Make hidden order
        self.hidden_order_color = copy.deepcopy(self.ball_colors)
        random.shuffle(self.hidden_order_color)
        #print("hidden order color: ", self.hidden_order_color)
        self.hidden_order_pos = []
        for color in self.hidden_order_color:
            self.hidden_order_pos.append(self.poses[self.ball_colors.index(color)])

        # initialization of visitation
        self.next_visit = 0
        self.reward_set = [1] * len(self.ball_colors)

        self.mission = "collect objects in hidden order as many as possible"

    def _reset_grid(self):
        # Create an empty grid
        self.grid = Grid(self.size, self.size)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, self.size, self.size)

        # Place the agent
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        # Place objects
        random.shuffle(self.ball_colors)
        for _pos, color in zip(self.poses, self.ball_colors):
            self.grid.set(*_pos, CollectableBall(color, 0))

        # Make hidden order
        self.hidden_order_pos = []
        for color in self.hidden_order_color:
            self.hidden_order_pos.append(self.poses[self.ball_colors.index(color)])

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        # Check if we hit a ball
        current_cell = self.grid.get(*self.agent_pos)
        agent_pos = tuple(self.agent_pos)
        if current_cell and current_cell.type == 'ball':
            if ((agent_pos[0] == self.hidden_order_pos[self.next_visit][0]) and
                    (agent_pos[1] == self.hidden_order_pos[self.next_visit][1])):
                self.grid.grid[agent_pos[1] * self.grid.width + agent_pos[0]] = None
                reward += self.reward_set[self.next_visit]
                self.reward_set[self.next_visit] = 0
                self.next_visit += 1
            else:
                self.next_visit = 0
                self._reset_grid()

        # Check if agent collects every ball in the order
        if self.next_visit >= len(self.ball_colors):
            self.next_visit = 0
            self.reward_set = [1] * len(self.ball_colors)
            self._reset_grid()

        reward -= self.step_penalty

        return obs, reward, done, info


class OrderMemoryN3Env(OrderMemoryEnv):
    def __init__(self, **kwargs):
        super().__init__(numObjs=3, **kwargs)

class OrderMemoryN3EnvWithPenalty(OrderMemoryEnv):
    def __init__(self, **kwargs):
        super().__init__(numObjs=3, step_penalty=0.05, **kwargs)

class OrderMemoryN4Env(OrderMemoryEnv):
    def __init__(self, **kwargs):
        super().__init__(numObjs=4, **kwargs)

class OrderMemoryN4EnvWithPenalty(OrderMemoryEnv):
    def __init__(self, **kwargs):
        super().__init__(numObjs=4, step_penalty=0.05, **kwargs)


register(
    id='MiniGrid-OrderMemory-N3-v0',
    entry_point='gym_minigrid.envs:OrderMemoryN3Env'
)

register(
    id='MiniGrid-OrderMemory-N3-Penalty-v0',
    entry_point='gym_minigrid.envs:OrderMemoryN3EnvWithPenalty'
)

register(
    id='MiniGrid-OrderMemory-N4-v0',
    entry_point='gym_minigrid.envs:OrderMemoryN4Env'
)

register(
    id='MiniGrid-OrderMemory-N4-Penalty-v0',
    entry_point='gym_minigrid.envs:OrderMemoryN4EnvWithPenalty'
)

