import random, copy
from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class OrderMemoryLargeEnv(MiniGridEnv):
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
        size=6,
        num_objs=3,
        area_size=2,
        step_penalty=0.0,
        agent_view_size=7,
        reset_positions=False,
    ):
        assert (size-2) % area_size == 0

        num_areas = ((size-2) // area_size)**2
        assert num_areas > num_objs

        #max_steps = 100 * area_size
        max_steps = 10 * size * num_objs**2
        self.num_areas = num_areas
        self.area_size = area_size
        self.num_objs = num_objs
        self.size = size
        self.max_steps = max_steps
        self.ball_colors = COLOR_NAMES[:num_objs]
        # start in center
        self.agent_area = num_areas // 2
        ball_areas = [x for x in range(num_areas) if x != self.agent_area]
        self.ball_areas = []
        for i in range(num_objs):
            self.ball_areas.append(ball_areas[i*((num_areas-1)//num_objs)])
        self.step_penalty = step_penalty
        self.reset_positions = reset_positions

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _get_coords_for_area(self, area):
        num_areas_per_row = (self.size-2) // self.area_size
        area_row = area // num_areas_per_row
        area_col = area % num_areas_per_row
        row_offset = area_row * self.area_size
        col_offset = area_col * self.area_size
        coords = []
        for i in range(self.area_size):
            for j in range(self.area_size):
                coords.append((i+col_offset+1, j+row_offset+1))
        return coords

    def _get_agent_pos(self):
        area = self.agent_area
        coords = self._get_coords_for_area(self.agent_area)
        agent_pos = random.choice(coords)
        return agent_pos

    def _get_poses(self):
        poses = []
        for ball_area in self.ball_areas:
            coords = self._get_coords_for_area(ball_area)
            coord = random.choice(coords)
            poses.append(coord)
        return poses

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent
        self.agent_pos = self.agent_start_pos = self._get_agent_pos()
        self.agent_dir = self.agent_start_dir = 3

        # Place objects
        self.poses = self._get_poses()
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
        if self.reset_positions:
            self.agent_pos = self._get_agent_pos()
            self.agent_dir = 3
        else:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir

        if self.reset_positions:
            self.poses = self._get_poses()
            random.shuffle(self.ball_colors)

        # Place objects
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
                obs = self.gen_obs()

        # Check if agent collects every ball in the order
        if self.next_visit >= len(self.ball_colors):
            self.next_visit = 0
            self.reward_set = [1] * len(self.ball_colors)
            self._reset_grid()

        reward -= self.step_penalty

        return obs, reward, done, info


class OrderMemoryLargeS6N3(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=3, agent_view_size=7, **kwargs)
register(
    id='MiniGrid-OrderMemoryLargeS6N3-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLargeS6N3'
)

class OrderMemoryLargeS9N3(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=11 because walls take up one so map will be 6x6
        super().__init__(size=11, area_size=3, num_objs=3, agent_view_size=7, **kwargs)
register(
    id='MiniGrid-OrderMemoryLargeS9N3-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLargeS9N3'
)

class OrderMemoryLargeS12N3(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=11 because walls take up one so map will be 6x6
        super().__init__(size=14, area_size=4, num_objs=3, agent_view_size=7, **kwargs)
register(
    id='MiniGrid-OrderMemoryLargeS12N3-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLargeS12N3'
)

class OrderMemoryLargeS15N3(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=11 because walls take up one so map will be 6x6
        super().__init__(size=17, area_size=5, num_objs=3, agent_view_size=7, **kwargs)
register(
    id='MiniGrid-OrderMemoryLargeS15N3-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLargeS15N3'
)


class OrderMemoryLargeS6N4(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=4, agent_view_size=7, **kwargs)
register(
    id='MiniGrid-OrderMemoryLargeS6N4-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLargeS6N4'
)

class OrderMemoryLargeS9N4(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=11 because walls take up one so map will be 6x6
        super().__init__(size=11, area_size=3, num_objs=4, agent_view_size=7, **kwargs)
register(
    id='MiniGrid-OrderMemoryLargeS9N4-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLargeS9N4'
)


class OrderMemoryLargeS6N5(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=5, agent_view_size=7, **kwargs)
register(
    id='MiniGrid-OrderMemoryLargeS6N5-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLargeS6N5'
)

class OrderMemoryLargeS9N5(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=11 because walls take up one so map will be 6x6
        super().__init__(size=11, area_size=3, num_objs=5, agent_view_size=7, **kwargs)
register(
    id='MiniGrid-OrderMemoryLargeS9N5-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLargeS9N5'
)


