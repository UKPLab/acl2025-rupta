from heapq import heappush, heappop
from typing import Callable, List, Set, TypeVar, Tuple, Any, Optional


State = TypeVar('State')
UniqueID = TypeVar('UniqueID')


def ucs(
    start: State,
    is_goal: Callable[[State], bool],
    expand: Callable[[State], Set[Tuple[State, float]]],
    get_unique_id: Callable[[State], UniqueID] = lambda x: x,
    when_none: Callable[[List[State]], Any] = lambda x: None,
) -> Optional[State]:
    """Lazy Uniform Cost Search.

    Args:
        start: start node
        is_goal: function to check if a node is the goal node
        expand: function to expand a node, returns a list of (child, cost) pairs
        get_unique_id (optional): function to get a unique id for a node
        when_none (optional): function to call when no path is found, gives all the visited nodes

    Returns:
        The goal node when found. Returns None if no path is found.
        Some implementations off expand will not ever return None.
    """
    # is the start node the goal node?
    if is_goal(start):
        return start

    # start up the queue
    queue = []

    # hack for unhashable types. Python is so well designed!
    class CmpFalse(object): __eq__ = __lt__ = __gt__ = lambda s,o: False
    heappush(queue, (0, CmpFalse(), start))
    all_visited: List[State] = [start]
    visited_dedup: set[UniqueID] = set()

    while not len(queue) == 0:
        cost, _, node = heappop(queue)
        visited_dedup.add(get_unique_id(node))
        all_visited.append(node)

        for child, child_cost in expand(node):
            # skip if already visited
            if get_unique_id(child) in visited_dedup:
                continue

            # check if it's goal
            if is_goal(child):
                return child

            # add to queue
            heappush(queue, (cost + child_cost, CmpFalse(), child))

    return when_none(all_visited)


if __name__ == '__main__':
    # define the graph
    graph = {
        'A': set([('B', 1), ('C', 2)]),
        'B': set([('D', 3), ('A', 0)]),
        'C': set([('D', 1)]),
        'D': set([('E', 1)]),
        'E': set(),
    }

    # define the goal
    goal = 'E'

    # define the expand function
    def expand(node: str) -> Set[Tuple[str, float]]:
        return graph[node]

    # define the is_goal function
    def is_goal(node: str) -> bool:
        return node == goal

    # run the search
    result = ucs('A', is_goal, expand)
    print(result)
