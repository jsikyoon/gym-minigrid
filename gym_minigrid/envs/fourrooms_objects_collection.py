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

    def __init__(self, num_good_obj=4, num_bad_obj=4, agent_pos=None, goal_pos=None):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        self.rand = None
        self.good_color = "yellow"
        self.bad_color = "blue"
        self.num_good_obj = num_good_obj
        self.num_bad_obj = num_bad_obj

        super().__init__(grid_size=17, max_steps=100)

    def _get_poses(self):
        total_coords = []
        for i in range(self.width):
            for j in range(self.height):
                if self.grid.get(i, j) is None:
                    total_coords.append((i, j))

        return total_coords

    def _reset_grid(self):

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

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())

        self.mission = "Reach the goal"

        self.obj_pos = self._get_poses()
        self.rand.shuffle(self.obj_pos)
        self.sampled_pos = self.rand.sample(
            self.obj_pos, self.num_bad_obj + self.num_good_obj
        )
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
            if current_cell.type == "ball" and current_cell.color == self.bad_color:
                self.grid.grid[agent_pos[1] * self.grid.width + agent_pos[0]] = None
                reward += -1.0
            if current_cell.type == "goal":
                reward += 3.0
                self._reset_grid()
                self.place_agent()

        if self.step_count >= self.max_steps:
            done = True

        return obs, reward, done, info


register(
    id="MiniGrid-FourRooms-Objects-v0",
    entry_point="gym_minigrid.envs:FourRoomsObjectsEnv",
)

