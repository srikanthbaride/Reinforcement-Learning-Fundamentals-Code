import numpy as np
from typing import List, Tuple

class GridWorld4x4:
    ACTIONS = ("up", "right", "down", "left")

    def __init__(self, step_reward=-1.0, goal_reward=0.0, deterministic=True,
                 self_transition_on_wall=True, goal=(0, 3)):
        assert deterministic, "Only deterministic supported."
        self.h, self.w = 4, 4
        self.step_reward, self.goal_reward = float(step_reward), float(goal_reward)
        self.self_transition_on_wall = self_transition_on_wall
        self.goal = tuple(goal)
        self._S = [(r, c) for r in range(self.h) for c in range(self.w)]
        self._A = list(self.ACTIONS)
        self._si = {s:i for i,s in enumerate(self._S)}
        self._ai = {a:i for i,a in enumerate(self._A)}
        self._P = np.zeros((len(self._S), len(self._A), len(self._S)))
        self._R = np.zeros_like(self._P)
        self._build_PR()

    def states(self): return list(self._S)
    def actions(self): return list(self._A)
    def state_index(self,r,c): return self._si[(r,c)]
    def action_index(self,a): return self._ai[a]
    def P_tensor(self): return self._P.copy()
    def R_tensor(self): return self._R.copy()

    def _in_bounds(self,r,c): return 0<=r<self.h and 0<=c<self.w
    def _next_state(self,s,a):
        if s==self.goal: return s
        r,c=s
        moves={"up":(-1,0),"right":(0,1),"down":(1,0),"left":(0,-1)}
        dr,dc=moves[a]; nr,nc=r+dr,c+dc
        if self._in_bounds(nr,nc): return (nr,nc)
        return s if self.self_transition_on_wall else s

    def _build_PR(self):
        for si,s in enumerate(self._S):
            for ai,a in enumerate(self._A):
                s2=self._next_state(s,a); s2i=self._si[s2]
                self._P[si,ai,s2i]=1.0
                r=self.step_reward if s2!=self.goal else self.step_reward
                self._R[si,ai,s2i]=r
