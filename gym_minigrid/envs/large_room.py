from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import random

class LargeRoomEnv(MiniGridEnv):
    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
        change_colors=False,
        seed=0,
        agent_view_size=7,
        obj_change_prob=0.05,
        explore_phase_steps=None,
    ):
        self.numObjs = size ** 2 // 5
        self.agent_start_dir = agent_start_dir
        self.agent_start_pos = agent_start_pos
        self.change_colors = change_colors
        self._seed = seed
        self.obj_change_prob = obj_change_prob
        self.agent_view_size = agent_view_size
        self.explore_phase_steps = explore_phase_steps
        if self.explore_phase_steps is not None:
            self.step_count = 0
            self.seen_obj_pos = set()
        change_groups = (
            ('red', 'green', 'blue'),
            ('purple', 'yellow', 'grey'),
            ('magenta', 'white', 'orange'),
        )
        self.change_group_map = {}
        for g in change_groups:
            for c in g:
                self.change_group_map[c] = g

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True,
            seed=seed,
            agent_view_size=agent_view_size,
        )

    def _gen_grid(self, width, height):
        self.rand = random.Random(self._seed)
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.objs = []

        # Place the agent
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        while len(self.objs) < int(self.numObjs):
            color = self.rand.choice(list(COLORS.keys()))
            obj = CollectableBall(color, 0)
            pos = self.place_obj(obj)
            self.objs.append(obj)


        self.mission = "walk in a circle"

        self.pos_cnt = 0
        if self.explore_phase_steps is not None:
            self.add_seen()

    def add_seen(self):
        for obj in self.objs:
            if self.in_view(*obj.cur_pos):
                self.seen_obj_pos.add(tuple(obj.cur_pos))

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        if self.change_colors:
            for idx, obj in enumerate(self.objs):
                if self.rand.random() > self.obj_change_prob:
                    continue

                self.grid.grid[obj.cur_pos[1] * self.grid.width + obj.cur_pos[0]] = None

                color = self.rand.choice(self.change_group_map[obj.color])
                new_obj = CollectableBall(color, 0)
                self.put_obj(new_obj, *obj.cur_pos)
                self.objs[idx] = new_obj

        if self.explore_phase_steps is not None:
            if self.step_count < self.explore_phase_steps:
                # explore phase, add seen objects to seen_objs
                self.add_seen()

            elif self.step_count == self.explore_phase_steps:
                # end of explore phase, remove unseen objs
                new_objs = []
                for obj in self.objs:
                    if tuple(obj.cur_pos) in self.seen_obj_pos:
                        new_objs.append(obj)
                    else:
                        self.grid.grid[obj.cur_pos[1] * self.grid.width + obj.cur_pos[0]] = None
                self.objs = new_objs

        return obs, reward, done, info


register(
    id='MiniGrid-LargeRoom-v0',
    entry_point='gym_minigrid.envs:LargeRoomEnv'
)


