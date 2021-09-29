import random, copy
from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import numpy as np


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
        num_key=2,
        reset_positions=False,
        agent_init_bottom=False,
        max_steps=None,
        wrong_reinit=False,
        dist_thr=5,
    ):
        assert (size-2) % area_size == 0

        num_areas = ((size-2) // area_size)**2
        assert num_areas > num_objs

        if max_steps is None:
            max_steps = 100 * area_size
        self.num_areas = num_areas
        self.area_size = area_size
        self.num_objs = num_objs
        self.size = size
        self.max_steps = max_steps
        self.ball_colors = COLOR_NAMES[:num_objs]
        self.agent_init_bottom = agent_init_bottom
        self.wrong_reinit = wrong_reinit
        # start in center
        if agent_init_bottom:
            self.agent_area = num_areas - (size - 2) // area_size // 2 - 1
        else:
            self.agent_area = num_areas // 2
        self.ball_areas = [x for x in range(num_areas) if x != self.agent_area]
        #self.ball_areas = []
        #for i in range(num_objs):
            #self.ball_areas.append(ball_areas[i*((num_areas-1)//num_objs)])
        self.step_penalty = step_penalty
        self.reset_positions = reset_positions
        self.dist_thr = dist_thr
        self.key_colors = COLOR_NAMES[num_objs:]
        self.poses = None
        self.num_key = num_key

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
        if self.agent_init_bottom:
            agent_pos = coords[-1]
        else:
            agent_pos = random.choice(coords)
        return agent_pos

    def _get_poses(self):
        total_coords = []
        for ball_area in self.ball_areas:
            total_coords = total_coords + self._get_coords_for_area(ball_area)
        random.shuffle(total_coords)
        coords = total_coords[:self.num_objs]
        coords_np = np.array(coords)
        dist_mat = abs(coords_np.reshape(1, self.num_objs, 2) - coords_np.reshape(self.num_objs, 1, 2)).sum(-1)
        dist_mat[np.arange(self.num_objs), np.arange(self.num_objs)] = 100
        min_dist = dist_mat.min()

        while min_dist <= self.dist_thr:
            random.shuffle(total_coords)
            coords = total_coords[:self.num_objs]
            coords_np = np.array(coords)
            dist_mat = abs(coords_np.reshape(1, self.num_objs, 2) - coords_np.reshape(self.num_objs, 1, 2)).sum(-1)
            dist_mat[np.arange(self.num_objs), np.arange(self.num_objs)] = 100
            min_dist = dist_mat.min()

        return coords

    def _get_key_poses(self):
        assert self.poses is not None

        total_coords = []
        for ball_area in self.ball_areas:
            total_coords = total_coords + self._get_coords_for_area(ball_area)
        random.shuffle(total_coords)
        coords = total_coords[:self.num_key]
        good_pos = True
        for coor in coords:
          if coor in self.poses:
            good_pos = False

        while not good_pos:
          random.shuffle(total_coords)
          coords = total_coords[:self.num_key]
          good_pos = True
          for coor in coords:
            if coor in self.poses:
              good_pos = False

        return coords

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

        # Place keys
        self.key_poses = self._get_key_poses()
        random.shuffle(self.key_colors)
        for _pos, color in zip(self.key_poses, self.key_colors[:self.num_key]):
            self.grid.set(*_pos, Key(color))

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

    def _reset_grid(self, reset_key=False):
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

        # Place keys
        if reset_key:
          self.key_poses = self._get_key_poses()
          random.shuffle(self.key_colors)
        for _pos, color in zip(self.key_poses, self.key_colors[:self.num_key]):
            self.grid.set(*_pos, Key(color))

        # Make hidden order
        self.hidden_order_pos = []
        for color in self.hidden_order_color:
            self.hidden_order_pos.append(self.poses[self.ball_colors.index(color)])

    def reset(self):
        obs = MiniGridEnv.reset(self)
        obs.update({
            'ball_pos': self.hidden_order_pos,
            'order': self.hidden_order_color,
            'next_visit': self.next_visit,
            })
        return obs

    def step(self, action):
        prev_agent_pos = self.agent_pos
        prev_agent_dir = self.agent_dir
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
                reward = -1.
                if not self.wrong_reinit:
                    self.agent_pos = prev_agent_pos
                    self.agent_dir = prev_agent_dir

        if current_cell and current_cell.type == 'key':
            self.grid.grid[agent_pos[1] * self.grid.width + agent_pos[0]] = None
            reward = 2.

        # Check if agent collects every ball in the order
        if self.next_visit >= len(self.ball_colors):
            self.next_visit = 0
            self.reward_set = [1] * len(self.ball_colors)
            self._reset_grid(reset_key=True)
            reward = 3.

        obs = self.gen_obs()
        reward -= self.step_penalty
        obs.update({
            'ball_pos': self.hidden_order_pos,
            'order': self.hidden_order_color,
            'next_visit': self.next_visit,
            })

        return obs, reward, done, info


class OrderMemoryLargeS6N3(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=3, agent_view_size=3, **kwargs)
register(
    id='MiniGrid-OrderMemoryLargeS6N3-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLargeS6N3'
)
class OrderMemoryLargeS6N3Penalty(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=3, step_penalty=0.05, agent_view_size=3, **kwargs)
register(
    id='MiniGrid-OrderMemoryLargeS6N3Penalty-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLargeS6N3Penalty'
)


class OrderMemoryLargeS6N4(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=4, agent_view_size=3, **kwargs)
register(
    id='MiniGrid-OrderMemoryLargeS6N4-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLargeS6N4'
)


class OrderMemoryLargeS6N4Penalty(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=4, step_penalty=0.05, agent_view_size=3, **kwargs)
register(
    id='MiniGrid-OrderMemoryLargeS6N4Penalty-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLargeS6N4Penalty'
)


class OrderMemoryLargeS6N5(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=5, agent_view_size=3, **kwargs)
register(
    id='MiniGrid-OrderMemoryLargeS6N5-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLargeS6N5'
)


class OrderMemoryLargeS6N5Penalty(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 6x6
        super().__init__(size=8, area_size=2, num_objs=5, step_penalty=0.05, agent_view_size=3, **kwargs)
register(
    id='MiniGrid-OrderMemoryLargeS6N5Penalty-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLargeS6N5Penalty'
)

class OrderMemoryLargeS7N4(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 7x7
        super().__init__(size=9, area_size=1, num_objs=4, agent_view_size=3, **kwargs)
register(
    id='MiniGrid-OrderMemoryLargeS7N4-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLargeS7N4'
)


class OrderMemoryLargeS7N5(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 7x7
        super().__init__(size=10, area_size=1, num_objs=5, agent_view_size=3, **kwargs)
register(
    id='MiniGrid-OrderMemoryLargeS7N5-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLargeS7N5'
)

class OrderMemoryLargeS8N4(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 8x8
        super().__init__(size=10, area_size=2, num_objs=4, agent_view_size=3, **kwargs)
register(
    id='MiniGrid-OrderMemoryLargeS8N4-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLargeS8N4'
)


class OrderMemoryLargeS8N5(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 8x8
        super().__init__(size=10, area_size=2, num_objs=5, agent_view_size=3, **kwargs)
register(
    id='MiniGrid-OrderMemoryLargeS8N5-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLargeS8N5'
)


class OrderMemoryLargeS9N4(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 8x8
        super().__init__(size=11, area_size=1, num_objs=4, agent_view_size=3, **kwargs)
register(
    id='MiniGrid-OrderMemoryLargeS9N4-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLargeS9N4'
)


class OrderMemoryLargeS9N5(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 8x8
        super().__init__(size=11, area_size=1, num_objs=5, agent_view_size=3, **kwargs)
register(
    id='MiniGrid-OrderMemoryLargeS9N5-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLargeS9N5'
)

class OrderMemoryLargeS10N4(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 10x10
        super().__init__(size=12, area_size=2, num_objs=4, agent_view_size=3, **kwargs)
register(
    id='MiniGrid-OrderMemoryLargeS10N4-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLargeS10N4'
)

class OrderMemoryLargeS11N4(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 11x11
        super().__init__(size=13, area_size=1, num_objs=4, agent_view_size=3, **kwargs)
register(
    id='MiniGrid-OrderMemoryLargeS11N4-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLargeS11N4'
)

class OrderMemoryLargeS13N4(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 13x13
        super().__init__(size=15, area_size=1, num_objs=4, agent_view_size=3, **kwargs)
register(
    id='MiniGrid-OrderMemoryLargeS13N4-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLargeS13N4'
)


class OrderMemoryLargeS10N5(OrderMemoryLargeEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 10x10
        super().__init__(size=12, area_size=2, num_objs=5, agent_view_size=3, **kwargs)
register(
    id='MiniGrid-OrderMemoryLargeS10N5-v0',
    entry_point='gym_minigrid.envs:OrderMemoryLargeS10N5'
)


