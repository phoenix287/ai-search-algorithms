import numpy as np
from HaifaEnv import HaifaEnv
from typing import List, Tuple
from collections import deque
import heapdict


class BFSGAgent():
    def __init__(self) -> None:
      self.env = None
        

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
      self.env = env
      env.reset()

      # node = (state, path, cost)
      node = (env.get_initial_state(), [], 0.0)

      #initializing Open and Close
      # OPEN = FIFO queue
      OPEN = deque([node])
      OPEN_states = {node[0]}
      # CLOSE: Set containing fully expanded nodes
      CLOSE = set()
      
      if (env.is_final_state(node[0]) is True):
        return [], 0.0, len(CLOSE)

      while OPEN:
        node_state, node_path, node_path_cost = OPEN.popleft()
        OPEN_states.remove(node_state)
        CLOSE.add(node_state)

        #iterate through successors sorted by action num
        for action, (succ_state, succ_cost, succ_terminated) in sorted(env.succ(node_state).items()):

          #skip invalid states
          if succ_state is None:
            continue

          #constructing the child node
          child_path = node_path + [action]
          child_path_cost = node_path_cost + succ_cost
          child = (succ_state, child_path, child_path_cost)
          
          #avoid processing states already expanded or currently in frontier
          if (succ_state not in CLOSE and succ_state not in OPEN_states):

            #goal test
            if (env.is_final_state(succ_state) is True):
              return child_path, child_path_cost, len(CLOSE)
            #add the child to frontier
            OPEN.append(child)
            OPEN_states.add(succ_state)

      return [], float('inf'), len(CLOSE)

        
def HHaifa(env, state):
  min_dist = float('inf')
  c_passway = 100.0

  if (env.is_final_state(state) is True):
    return 0.0
  
  row, col = env.to_row_col(state)
  for goal in env.goals:
    goal_row, goal_col = env.to_row_col(goal)
    dist = abs(goal_row - row) + abs(goal_col - col)
    if dist < min_dist:
      min_dist = dist
  
  return min(min_dist, c_passway)


class GreedyAgent():
  
    def __init__(self) -> None:
        self.env = None

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        self.env = env
        env.reset()

        expanded = 0

        #initializing open and close
        #OPEN : priority queue sorted by hhaifa and state_id as tie breaker
        OPEN = heapdict.heapdict()
        # CLOSE: Set containing fully expanded nodes
        CLOSE = set()

        #node in OPEN = (hval, (state, path, cost))
        #initial state
        start_state = env.get_initial_state()
        start_node = (start_state, (), 0.0)
        OPEN[start_node] = (HHaifa(env, start_state), start_state)
        OPEN_states = {start_state}

        while OPEN:
          node, hval = OPEN.popitem()
          node_state, node_path, node_path_cost = node

          OPEN_states.remove(node_state)
          CLOSE.add(node_state)

          if (env.is_final_state(node_state)):
            return list(node_path), node_path_cost, expanded
          
          #expand after the goal test
          expanded += 1

          for action, (succ_state, succ_cost, succ_terminated) in env.succ(node_state).items():

            #skip if state is invalid
            if succ_state is None:
              continue

            #construct the child state
            child_path = node_path + (action,)
            child_path_cost = node_path_cost + succ_cost
            child = (succ_state, child_path, child_path_cost)

            #avoid processing states already expanded or currently in frontier
            if (succ_state not in CLOSE and succ_state not in OPEN_states):
              #add the child to frontier
              child_hval = HHaifa(env, succ_state)
              OPEN[child] = (child_hval, succ_state)
              OPEN_states.add(succ_state)
        
        return [], float('inf'), len(CLOSE)
          
          





class AStarEpsilonAgent():
    def __init__(self):
        self.env = None
    
    def min_manhattan_distance(self, state: int) -> float:
        goals = (self.env).goals
        if(state in goals):
           return 0.0
        min_d = float('inf')
        s_row, s_col = (self.env).to_row_col( state)
        for g in goals:
            g_row, g_col = (self.env).to_row_col( g)
            curr_d = abs(s_row-g_row)+abs(s_col-g_col)
            min_d = min(min_d, curr_d)
            
        return min_d

    def get_next_state(self, OPEN: heapdict) -> Tuple:
        # find f_min value
        f_min_node, f_min = OPEN.peekitem()
        # build focal list
        focal_list = [(idx, n) for idx, n in enumerate(OPEN) if (n[3] <= (1 + self.epsilon) * f_min)]
        # non deterministic approach
        next_node = min(focal_list, key=lambda item: ((item[1][3] - item[1][2]), item[0]))
        return next_node[1]
        
        # deterministic aporoach - return minimal node according to h_focal first and id second
        #return min(focal_list, key=lambda node:((node[3]-node[2]), node[0]))
    
    def h_focal(self, state: int) -> float: # heuristic for focal list (you don't have to use it)
        min_MD = self.min_manhattan_distance(state) 
        # C_passaway = ((self.env).nL_cost).getdict(b"P")
        return min(min_MD, 100.0)

    def search(self, env: HaifaEnv, epsilon: float = None)->Tuple[list[int],float,int]: 
        # initialize environment and epsilon
        self.env = env
        env.reset()
        self.epsilon = epsilon
        # start from an initial state
        initial_s = env.get_initial_state() 
        # node = (state id, path, g value, f value = g+h)
        initial_s_h_val = self.h_focal(initial_s)
        node = (initial_s, (), 0, initial_s_h_val) 
        #OPEN is heap dict: {node: f_value}, and OPEN_states, CLOSE are dicts: {state id: node}
        OPEN = heapdict.heapdict()
        OPEN[node] = initial_s_h_val
        OPEN_states = {initial_s: node}
        CLOSE={}
        # initialize number of expanded nodes with 0
        expanded = 0
        if(self.env.is_final_state(node[0])):
            return [], 0.0, expanded
        
        while OPEN:
            # pop node to expand according to a focal list 
           exp_node = self.get_next_state(OPEN)
           del OPEN[exp_node]
           del OPEN_states[exp_node[0]]
           if(self.env.is_final_state(exp_node[0])):
            return list(exp_node[1]), exp_node[2], expanded
           CLOSE[exp_node[0]] = exp_node
           
           # create/move successors based on whether they belong to OPEN/CLOSE or not
           for action,succ in sorted(self.env.succ(exp_node[0]).items()):
            if succ is None:
                continue
            succ_state = succ[0]
            g_value = exp_node[2]+succ[1]
            if (succ_state not in OPEN_states and succ_state not in CLOSE):
                f_value = g_value+self.h_focal(succ_state)
                s_node = (succ_state, exp_node[1]+(action,), g_value, f_value)
                OPEN[s_node] = f_value
                OPEN_states[succ_state] = s_node
            elif succ_state in OPEN_states:
                old_node = OPEN_states[succ_state]
                old_g_value = old_node[2]
                # move succesor if theres an improved path
                if(g_value < old_g_value):
                   new_f_value = old_node[3]-old_g_value+g_value
                   new_node = (succ_state, exp_node[1]+(action,),g_value,new_f_value)
                   del OPEN[old_node]
                   OPEN[new_node] = new_f_value
                   OPEN_states[succ_state] = new_node
            else:
               old_node = CLOSE[succ_state]
               old_g_value = old_node[2]
               # move succesor if theres an improved path
               if (g_value < old_g_value):
                new_f_value = old_node[3]-old_g_value+g_value
                new_node = (succ_state, exp_node[1]+(action,),g_value,new_f_value)
                OPEN[new_node] = new_f_value
                OPEN_states[succ_state] = new_node
                del CLOSE[succ_state]
           # update after calling succ!
           expanded+=1
        # no path to a goal state was found
        return [], float('inf'), expanded
               
            
                
            



class AStarAgent():
    
    def __init__(self):
        self.env = None


    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        self.env = env
        env.reset()
        # use A*-epsilon with epsilon = 0
        AStarEpsilon_agent = AStarEpsilonAgent()
        return AStarEpsilon_agent.search(env, 0)

