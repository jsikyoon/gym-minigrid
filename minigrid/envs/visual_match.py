from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import VM_Fruit, VM_Goal, VM_WrongGoal, Ball
from minigrid.minigrid_env import MiniGridEnv

DEFAULT_MAX_FRAMES_PER_PHASE = {
    "explore": 5,
    "distractor": 10,
    "reward": 15,
}
class VisualMatchEnv(MiniGridEnv):
    """
    ## Description
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 4), # to see the ball in the first room
        agent_start_dir=0,
        max_steps: int | None = None,
        active: bool = False,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        #if max_steps is None:
        #    max_steps = 4 * size**2
        max_steps = DEFAULT_MAX_FRAMES_PER_PHASE['explore'] + DEFAULT_MAX_FRAMES_PER_PHASE['distractor'] + DEFAULT_MAX_FRAMES_PER_PHASE['reward']
            
        self._size = size # use for reward phase

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "remember the ball color in first room, and collect the same color ball in third room"

    def _gen_grid(self, width, height): # it is explore phase grid
        # Create an empty grid
        self.width = width = self.height = height = self._size
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place objects
        self.goal_color = self._rand_elem(['green', 'blue', 'red'])
        self.grid.set(width//2, height//2, Ball(self.goal_color))
        
        ## Place a goal square in the bottom-right corner
        #self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self._phase = "explore"
        self.mission = "remember the ball color"
        
    def _gen_disctractor_grid(self):

        self.width = self.height = 30
        self.grid = Grid(self.width, self.height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, self.height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(self.width - 1, 0)
        
        self.place_agent()
        
        for _ in range(30):
            object = VM_Fruit('yellow')
            self.place_obj(object)

        self._phase = "distractor"
        self.mission = "collect yellow balls as many as possible"
        
    def _gen_reward_grid(self):
        # Create an empty grid
        width = height = self._size
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place objects
        objects_info = [
            ['green', [width//2, 1]],
            ['blue', [width//2, height//2]],
            ['red', [width//2, height-2]],
        ]
        for _obj_info in objects_info:
            color, pos = _obj_info
            if color == self.goal_color:
                self.grid.set(pos[0], pos[1], VM_Goal(color)) # goal ball
            else:
                self.grid.set(pos[0], pos[1], VM_WrongGoal(color)) # distractor ball
        
        ## Place a goal square in the bottom-right corner
        #self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self._phase = "reward"
        self.mission = "collect the same colored ball with that in the first room"

    def step(self, action):
        obs, reward, terminated, truncated, info = MiniGridEnv.step(self, action)
        # move to the next phase
        if self._phase == "explore" and self.step_count > DEFAULT_MAX_FRAMES_PER_PHASE[self._phase]:
            self._gen_disctractor_grid()
        if self._phase == "distractor" and self.step_count > DEFAULT_MAX_FRAMES_PER_PHASE[self._phase] + DEFAULT_MAX_FRAMES_PER_PHASE["explore"]:
            self._gen_reward_grid()
        obs = self.gen_obs()

        return obs, reward, terminated, truncated, info       