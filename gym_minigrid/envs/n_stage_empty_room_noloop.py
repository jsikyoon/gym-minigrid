from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import random


class NStageEmptyEnvNoLoop(MiniGridEnv):
    """
    N-stage empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=6,
        num_stages=4,
        agent_start_pos=None,
        agent_start_dir=0,
        stage_one_period=10,
        max_steps=100,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        self.stage_idx = 0
        self.num_stages = num_stages
        self.ball_colors = COLOR_NAMES[:num_stages]
        self.next_visit = 0
        self.rand = None
        self.stage_one_period = stage_one_period
        self.dist_thr = 2.0

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=True,
        )

    def _get_poses(self):
        total_coords = []
        for i in range(self.width):
            for j in range(self.height):
                if self.grid.get(i, j) is None:
                    total_coords.append((i, j))

        return total_coords

    def _reset_grid(self, reset_key=False):
        self.stage_idx = 0
        self.next_visit = 0
        self._set_stage()

    def reset(self):
        self.stage_idx = 0
        self.next_visit = 0
        self.stay_time = 0
        self.frist_full_stage = True
        obs = MiniGridEnv.reset(self)
        return obs

    def _gen_grid(self, width, height):
        if self.rand is None:
            self.rand = random.Random(self.seed)

        self.grid = Grid(self.width, self.height)  # to get position list
        self.grid.wall_rect(0, 0, self.width, self.height)
        obj_pos = self._get_poses()
        self.rand.shuffle(obj_pos)
        min_dist = 0.0
        while min_dist < self.dist_thr:
            sampled_pos = self.rand.sample(
                obj_pos, self.num_stages + 1
            )  # one for agent
            sampled_pos_np = np.array(sampled_pos)
            dist_mat = abs(
                sampled_pos_np.reshape(1, self.num_stages + 1, 2)
                - sampled_pos_np.reshape(self.num_stages + 1, 1, 2)
            ).sum(-1)
            dist_mat[
                np.arange(self.num_stages + 1), np.arange(self.num_stages + 1),
            ] = 100
            min_dist = dist_mat.min()

        self.sampled_pos = sampled_pos[: self.num_stages]
        agent_pos = sampled_pos[-1]
        self.rand.shuffle(self.ball_colors)
        # begin first stage
        self._set_stage()

        # Place the agent
        self.agent_pos = agent_pos
        self.agent_dir = self._rand_int(0, 4)

        self.agent_init_pos = agent_pos
        self.agent_init_dir = self.agent_dir

        self.mission = "get to the green goal square"

    def _set_stage(self, reset_agent=False):

        # Create an empty grid
        self.grid = Grid(self.width, self.height)
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, self.width, self.height)

        if self.stage_idx != self.num_stages:
            ball_color = self.ball_colors[self.stage_idx]
            self.grid.set(
                *self.sampled_pos[self.stage_idx], CollectableBall(ball_color, 0)
            )

        else:
            for i in range(self.num_stages):
                ball_color = self.ball_colors[i]
                self.grid.set(*self.sampled_pos[i], CollectableBall(ball_color, 0))
            if self.frist_full_stage:
                self.frist_full_stage = False
                self.agent_pos = self.agent_init_pos
                self.agent_dir = self.agent_init_dir

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        reward = 0  # rewrite the reward logic
        done = False

        current_cell = self.grid.get(*self.agent_pos)
        agent_pos = tuple(self.agent_pos)
        self.stay_time += 1
        if current_cell:
            if current_cell.type == "ball" and (self.stage_idx != self.num_stages):
                self.grid.grid[agent_pos[1] * self.grid.width + agent_pos[0]] = None
                reward += 1.0
                self.stage_idx += 1
                self._set_stage()
                self.stay_time = 0
            elif current_cell.type == "ball" and (self.stage_idx == self.num_stages):
                if (agent_pos[0] == self.sampled_pos[self.next_visit][0]) and (
                    agent_pos[1] == self.sampled_pos[self.next_visit][1]
                ):
                    self.grid.grid[agent_pos[1] * self.grid.width + agent_pos[0]] = None
                    self.next_visit += 1
                    reward += 2  # to be distinguished from stage 1 reward (for logs)
                    if self.next_visit == self.num_stages:
                        reward += 1.0  # +3 reward for complete a circle
                        done = True
                else:
                    reward += -1.0
                    done = True
        else:
            if (self.stay_time >= self.stage_one_period) and (
                self.stage_idx != self.num_stages
            ):
                self.stage_idx += 1
                self.stage_idx = min(self.stage_idx, self.num_stages)
                self._set_stage()
                self.stay_time = 0

        obs = self.gen_obs()
        if self.step_count >= self.max_steps:
            done = True

        return obs, reward, done, info


class NStageEmptyEnvNoLoopS7(NStageEmptyEnvNoLoop):
    def __init__(self, **kwargs):
        super().__init__(size=7, **kwargs)


register(
    id="MiniGrid-NStageEmptyEnvNoLoopS7-v0",
    entry_point="gym_minigrid.envs:NStageEmptyEnvNoLoopS7",
)
