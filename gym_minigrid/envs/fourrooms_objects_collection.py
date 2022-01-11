#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import random


class FourRoomsObjectsEnv(MiniGridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(
        self,
        num_good_obj=4,
        num_bad_obj=4,
        grid_size=17,
        yellow_first=False,
        agent_pos=None,
        goal_pos=None,
        max_step=100,
    ):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        self.rand = None
        self.good_color = "yellow"
        self.bad_color = "blue"
        self.num_good_obj = num_good_obj
        self.num_bad_obj = num_bad_obj
        self.dist_thr = 2
        self.yellow_first = yellow_first
        self.yellow_colleted = 0
        self.agent_goal_dist_thr = 20
        self.goal_colors = ["red", "green"]

        super().__init__(grid_size=grid_size, max_steps=max_step)

    def _get_poses(self):
        total_coords = []
        for i in range(self.width):
            for j in range(self.height):
                if self.grid.get(i, j) is None:
                    total_coords.append((i, j))

        return total_coords

    def _reset_grid(self):
        self.yellow_colleted = 0

        # objects reappear
        for i in range(self.num_good_obj):
            self.grid.set(*self.sampled_pos[i], CollectableBall(self.good_color, 0))
        for i in range(self.num_good_obj, self.num_bad_obj + self.num_good_obj):
            self.grid.set(*self.sampled_pos[i], CollectableBall(self.bad_color, 0))

    def _gen_grid(self, width, height):
        if self.rand is None:
            self.rand = random.Random(self.seed)
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)

        obj_pos = self._get_poses()
        self.rand.shuffle(obj_pos)
        # sample goal and agent
        sampled_pos = self.rand.sample(obj_pos, 3)
        sampled_pos_np = np.array(sampled_pos)
        dist_mat = abs(
            sampled_pos_np.reshape(1, 3, 2) - sampled_pos_np.reshape(3, 1, 2)
        ).sum(-1)
        dist_mat[np.arange(3), np.arange(3),] = 100

        min_dist = dist_mat.min()
        while min_dist <= self.agent_goal_dist_thr:
            sampled_pos = self.rand.sample(obj_pos, 3)
            sampled_pos_np = np.array(sampled_pos)
            dist_mat = abs(
                sampled_pos_np.reshape(1, 3, 2) - sampled_pos_np.reshape(3, 1, 2)
            ).sum(-1)
            dist_mat[np.arange(3), np.arange(3),] = 100
            min_dist = dist_mat.min()

        agent_pos = sampled_pos[0]
        goal_clue_pos = list(agent_pos)
        if agent_pos[0] <= 2:
            agent_dir = 0
        elif agent_pos[0] >= 19:
            agent_dir = 2
        elif agent_pos[1] <= 2:
            agent_dir = 1
        elif agent_pos[1] >= 19:
            agent_dir = 3
        else:
            agent_dir = self.rand.sample(range(4), 1)[0]

        if agent_dir == 0:  # look right
            goal_clue_pos[0] += 2
        elif agent_dir == 1:  # look down
            goal_clue_pos[1] += 2
        elif agent_dir == 2:  # look left
            goal_clue_pos[0] -= 2
        elif agent_dir == 3:  # look up
            goal_clue_pos[1] -= 2

        self.agent_pos = agent_pos
        self.agent_dir = agent_dir

        true_goal_pos = sampled_pos[1]
        false_goal_pos = sampled_pos[2]
        true_color_idx = self.rand.sample(range(2), 1)[0]
        false_color_idx = 1 - true_color_idx
        goal_clue = Goal(color=self.goal_colors[true_color_idx])
        goal_clue.set_name("goal_clue")
        self.put_obj(goal_clue, *goal_clue_pos)
        true_goal = Goal(color=self.goal_colors[true_color_idx])
        true_goal.set_name("true_goal")
        self.put_obj(true_goal, *true_goal_pos)
        false_goal = Goal(color=self.goal_colors[false_color_idx])
        false_goal.set_name("false_goal")
        self.put_obj(false_goal, *false_goal_pos)

        self.mission = "Reach the goal"

        # place distractors
        obj_pos = self._get_poses()
        self.rand.shuffle(obj_pos)
        sampled_pos = self.rand.sample(obj_pos, self.num_bad_obj + self.num_good_obj)
        sampled_pos_np = np.array(sampled_pos)
        dist_mat = abs(
            sampled_pos_np.reshape(1, self.num_bad_obj + self.num_good_obj, 2)
            - sampled_pos_np.reshape(self.num_bad_obj + self.num_good_obj, 1, 2)
        ).sum(-1)
        dist_mat[
            np.arange(self.num_bad_obj + self.num_good_obj),
            np.arange(self.num_bad_obj + self.num_good_obj),
        ] = 100
        min_dist = dist_mat.min()

        while min_dist <= self.dist_thr:
            sampled_pos = self.rand.sample(
                obj_pos, self.num_bad_obj + self.num_good_obj
            )
            sampled_pos_np = np.array(sampled_pos)
            dist_mat = abs(
                sampled_pos_np.reshape(1, self.num_bad_obj + self.num_good_obj, 2)
                - sampled_pos_np.reshape(self.num_bad_obj + self.num_good_obj, 1, 2)
            ).sum(-1)
            dist_mat[
                np.arange(self.num_bad_obj + self.num_good_obj),
                np.arange(self.num_bad_obj + self.num_good_obj),
            ] = 100
            min_dist = dist_mat.min()

        self.sampled_pos = sampled_pos

        for i in range(self.num_good_obj):
            self.grid.set(*self.sampled_pos[i], CollectableBall(self.good_color, 0))
        for i in range(self.num_good_obj, self.num_bad_obj + self.num_good_obj):
            self.grid.set(*self.sampled_pos[i], CollectableBall(self.bad_color, 0))

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        reward = 0  # rewrite the reward logic
        done = False

        current_cell = self.grid.get(*self.agent_pos)
        agent_pos = tuple(self.agent_pos)
        if current_cell:
            if current_cell.type == "ball" and current_cell.color == self.good_color:
                self.grid.grid[agent_pos[1] * self.grid.width + agent_pos[0]] = None
                reward += 1.0
                self.yellow_colleted += 1
            if current_cell.type == "ball" and current_cell.color == self.bad_color:
                self.grid.grid[agent_pos[1] * self.grid.width + agent_pos[0]] = None
                reward += -1.0
            if current_cell.type == "goal":
                if (not self.yellow_first) or (
                    self.yellow_colleted == self.num_good_obj
                ):
                    if current_cell.name == "goal_clue":
                        self.grid.grid[
                            agent_pos[1] * self.grid.width + agent_pos[0]
                        ] = None
                        reward += 1.0
                    if current_cell.name == "true_goal":
                        reward += 3.0
                        self.place_agent()
                    if current_cell.name == "false_goal":
                        reward -= 1.0
                        self.place_agent()
                    # self._reset_grid()

        obs = self.gen_obs()
        if self.step_count >= self.max_steps:
            done = True

        return obs, reward, done, info


register(
    id="MiniGrid-FourRooms-Objects-v0",
    entry_point="gym_minigrid.envs:FourRoomsObjectsEnv",
)


class FourRoomsObjectsEnvS23N8(FourRoomsObjectsEnv):
    def __init__(self, **kwargs):
        # size=8 because walls take up one so map will be 5x5
        super().__init__(num_good_obj=8, num_bad_obj=8, grid_size=23, **kwargs)


register(
    id="MiniGrid-FourRooms-ObjectsEnvS23N8-v0",
    entry_point="gym_minigrid.envs:FourRoomsObjectsEnvS23N8",
)


class FourRoomsObjectsEnvS11N4YellowFirst(FourRoomsObjectsEnv):
    def __init__(self, **kwargs):
        # room size=4, yellow_first
        super().__init__(
            num_good_obj=4, num_bad_obj=4, grid_size=11, yellow_first=True, **kwargs
        )


register(
    id="MiniGrid-FourRooms-ObjectsEnvS11N4YellowFirst-v0",
    entry_point="gym_minigrid.envs:FourRoomsObjectsEnvS11N4YellowFirst",
)


class FourRoomsObjectsEnvS11N4(FourRoomsObjectsEnv):
    def __init__(self, **kwargs):
        # room size=4
        super().__init__(
            num_good_obj=4, num_bad_obj=4, grid_size=11, yellow_first=False, **kwargs
        )


register(
    id="MiniGrid-FourRooms-ObjectsEnvS11N4-v0",
    entry_point="gym_minigrid.envs:FourRoomsObjectsEnvS11N4",
)
