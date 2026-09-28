# ai-search-algorithms

Python implementation of  AI search algorithms: BFS, Greedy, A*-epsilon and A*. Written for Intro to AI course.

## The task
for each algorithm, and given the course's HaifaEnv file, find the algorithm's path from initial state to goal state and return the tuple of (path as a list of states, path's cost, number of expanded nodes).

## Algorithms

- **BFS**: find path with fewest steps.
- **Greedy**: find path by finding optimal move at each step, optimal decided by minimal value of course's suggested huritstic namely HHaifa.
- **A*-epsilon**: find a path of up to (1+epsilon)*cost of optimal path.
-**A\***: find optimal path using Manhattan distance huristic.

## Notes
- Requires `heapdict` (`pip install -r requirements.txt`).
- Needs the course's `HaifaEnv` environment, which isn't included.

## Credits
built with Sarah Shaheen for intro to AI course, semester Spring 2025-2026. 
