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
        agent_view_size=3,
        reset_positions=False,
        agent_bottom_start=False,
    ):
        assert (size-2) % area_size == 0

        num_areas = ((size-2) // area_size)**2
        assert num_areas > num_objs

        max_steps = 100 * area_size
        self.num_areas = num_areas
        self.area_size = area_size
        self.num_objs = num_objs
        self.size = size
        self.max_steps = max_steps
        self.ball_colors = COLOR_NAMES[:num_objs]
        if agent_bottom_start:
            # start at the bottom center
            self.agent_area = num_areas - (((size-2) // area_size) // 2) - 1
        else:
            # start in center
            self.agent_area = num_areas // 2
        ball_areas = [x for x in range(num_areas) if x != self.agent_area]
        random.shuffle(ball_areas)
        self.ball_areas = []
        for i in range(num_objs):
            self.ball_areas.append(ball_areas[i*((num_areas-1)//num_objs)])
        self.step_penalty = step_penalty
        self.reset_positions = reset_positions
        self.agent_bottom_start = agent_bottom_start

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
        if self.agent_bottom_start:
            return ((self.size-2) // 2, (self.size-2))
        else:
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

        # Check if agent collects every ball in the order
        if self.next_visit >= len(self.ball_colors):
            self.next_visit = 0
            self.reward_set = [1] * len(self.ball_colors)
            self._reset_grid()

        reward -= self.step_penalty

        return obs, reward, done, info

class OrderMemoryLarge6x6N3EnvWithPenalty(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=3, step_penalty=0.05, agent_view_size=3, **kwargs)

class OrderMemoryLarge6x6N3EnvWithPenaltyReset(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=3, step_penalty=0.05, agent_view_size=3, reset_positions=True, **kwargs)

class OrderMemoryLarge6x6N3EnvReset(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=3, agent_view_size=3, reset_positions=True, **kwargs)

class OrderMemoryLarge6x6N4EnvReset(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=4, agent_view_size=3, reset_positions=True, **kwargs)

class OrderMemoryLarge6x6N5EnvReset(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=5, agent_view_size=3, reset_positions=True, **kwargs)

class OrderMemoryLarge6x6N3EnvFixed(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=3, agent_view_size=7, agent_bottom_start=True, **kwargs)

class OrderMemoryLarge6x6N4EnvFixed(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=4, agent_view_size=7, agent_bottom_start=True,  **kwargs)

class OrderMemoryLarge6x6N5EnvFixed(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=5, agent_view_size=7, agent_bottom_start=True, **kwargs)

class OrderMemoryLarge6x6N3EnvResetFixed(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=3, agent_view_size=7, reset_positions=True, agent_bottom_start=True, **kwargs)

class OrderMemoryLarge6x6N4EnvResetFixed(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=4, agent_view_size=7, reset_positions=True, agent_bottom_start=True,  **kwargs)

class OrderMemoryLarge6x6N5EnvResetFixed(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=5, agent_view_size=7, reset_positions=True, agent_bottom_start=True, **kwargs)

class OrderMemoryLarge9x9N3EnvWithPenalty(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=11 because walls take up one so map will be 11x11
        super().__init__(size=11, area_size=3, num_objs=3, step_penalty=0.05, agent_view_size=3, **kwargs)

class OrderMemoryLarge6x6N4EnvWithPenalty(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=4, step_penalty=0.05, agent_view_size=3, **kwargs)

class OrderMemoryLarge6x6N5EnvWithPenalty(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=5, step_penalty=0.05, agent_view_size=3, **kwargs)

class OrderMemoryLarge6x6N6EnvWithPenalty(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=6, step_penalty=0.05, agent_view_size=3, **kwargs)

class OrderMemoryLarge6x6N7EnvWithPenalty(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=7, step_penalty=0.05, agent_view_size=3, **kwargs)

class OrderMemoryLarge6x6N8EnvWithPenalty(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=8, step_penalty=0.05, agent_view_size=3, **kwargs)

class OrderMemoryLarge9x9N4EnvWithPenalty(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=11 because walls take up one so map will be 11x11
        super().__init__(size=11, area_size=3, num_objs=4, step_penalty=0.05, agent_view_size=3, **kwargs)




register(
    id='MiniGrid-OrderMemoryLarge-N3-6x6-Penalty-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLarge6x6N3EnvWithPenalty'
)

register(
    id='MiniGrid-OrderMemoryLarge-N3-6x6-Penalty-Reset-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLarge6x6N3EnvWithPenaltyReset'
)

register(
    id='MiniGrid-OrderMemoryLarge-N3-6x6-Reset-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLarge6x6N3EnvReset'
)

register(
    id='MiniGrid-OrderMemoryLarge-N4-6x6-Reset-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLarge6x6N4EnvReset'
)

register(
    id='MiniGrid-OrderMemoryLarge-N5-6x6-Reset-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLarge6x6N5EnvReset'
)

register(
    id='MiniGrid-OrderMemoryLarge-N3-6x6-Fixed-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLarge6x6N3EnvFixed'
)

register(
    id='MiniGrid-OrderMemoryLarge-N4-6x6-Fixed-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLarge6x6N4EnvFixed'
)

register(
    id='MiniGrid-OrderMemoryLarge-N5-6x6-Fixed-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLarge6x6N5EnvFixed'
)


register(
    id='MiniGrid-OrderMemoryLarge-N3-6x6-Reset-Fixed-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLarge6x6N3EnvResetFixed'
)

register(
    id='MiniGrid-OrderMemoryLarge-N4-6x6-Reset-Fixed-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLarge6x6N4EnvResetFixed'
)

register(
    id='MiniGrid-OrderMemoryLarge-N5-6x6-Reset-Fixed-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLarge6x6N5EnvResetFixed'
)

register(
    id='MiniGrid-OrderMemoryLarge-N4-6x6-Penalty-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLarge6x6N4EnvWithPenalty'
)

register(
    id='MiniGrid-OrderMemoryLarge-N5-6x6-Penalty-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLarge6x6N5EnvWithPenalty'
)

register(
    id='MiniGrid-OrderMemoryLarge-N6-6x6-Penalty-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLarge6x6N6EnvWithPenalty'
)

register(
    id='MiniGrid-OrderMemoryLarge-N7-6x6-Penalty-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLarge6x6N7EnvWithPenalty'
)

register(
    id='MiniGrid-OrderMemoryLarge-N8-6x6-Penalty-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLarge6x6N8EnvWithPenalty'
)

register(
    id='MiniGrid-OrderMemoryLarge-N3-9x9-Penalty-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLarge9x9N3EnvWithPenalty'
)


register(
    id='MiniGrid-OrderMemoryLarge-N4-9x9-Penalty-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLarge9x9N4EnvWithPenalty'
)
