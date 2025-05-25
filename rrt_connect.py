import numpy as np
import mujoco


# don't count collisions between objects and table
DEFAULT_COLLISION_COUNT = 16


class Node:
    def __init__(self, q, parent=None):
        """
        Node class for RRTConnect.

        Args:
            q: joint configuration
            parent: parent node
        """
        self.q = q
        self.parent = parent

class RRTConnect:
    def __init__(self, model, data, dof_ids, step_size=0.05, max_iters=1000):
        """
        RRTConnect motion planner class.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            dof_ids: IDs of the dof to control
            step_size: step size for RRT
            max_iters: maximum number of iterations
        """
        self.model = model
        self.data = data
        self.dof_ids = dof_ids
        self.step_size = step_size
        self.max_iters = max_iters

        # RRT 
        self.start = None
        self.goal = None
        self.tree_a = []
        self.tree_b = []

        self.jnt_range = model.jnt_range[dof_ids].T

    def random_config(self):
        """Randomly sample a configuration from the joint range."""
        return np.random.uniform(*self.jnt_range)
    
    def is_valid(self, q):
        """Check if a configuration is valid."""
        self.data.qpos[self.dof_ids] = q
        mujoco.mj_forward(self.model, self.data)
        return self.data.ncon <= DEFAULT_COLLISION_COUNT
    
    def new_config(self, q_near, q_rand):
        """Generate a new configuration by interpolating between q_near and q_rand."""
        direction = q_rand - q_near
        length = np.linalg.norm(direction)
        q_new = None
        if length < self.step_size:
            q_new = q_rand
        else:
            q_new = q_near + self.step_size * direction / length
        return q_new, self.is_valid(q_new)
    
    def nearest_neighbor(self, tree, q):
        """Find the nearest neighbor node in the tree to the given configuration."""
        distances = np.linalg.norm(np.vstack([node.q for node in tree]) - q, axis=1)
        return tree[np.argmin(distances)]
    
    def extend(self, tree, q):
        """Extend the tree towards the given configuration."""
        node_near = self.nearest_neighbor(tree, q)
        q_near = node_near.q
        q_new, valid = self.new_config(q_near, q)
        if valid:
            node_new = Node(q_new, node_near)
            tree.append(node_new)
            return node_new
        return None
    
    def connect(self, tree, q):
        """Connect the tree to the given configuration."""
        # instead of repeatedly calling extend, use insight that nearest neighbor
        # is always the new node after the first call 
        node_near = self.nearest_neighbor(tree, q)
        q_near = node_near.q
        while True:
            q_new, valid = self.new_config(q_near, q)
            if not valid:
                return None 
            node_near = Node(q_new, node_near)
            q_near = node_near.q
            tree.append(node_near)
            if np.linalg.norm(q_near - q) <= self.step_size:
                return node_near
    
    def plan(self, q_start, q_goal):
        """Plan a path from q_start to q_goal."""
        self.start = Node(q_start)
        self.goal = Node(q_goal)
        self.tree_a = [self.start]
        self.tree_b = [self.goal]

        for _ in range(self.max_iters):
            q_rand = self.random_config()
            extended_node = self.extend(self.tree_a, q_rand)
            if extended_node is not None:
                connected_node = self.connect(self.tree_b, extended_node.q)
                if connected_node is not None:
                    # path found 
                    path_a = self.dfs(extended_node)
                    path_b = self.dfs(connected_node)
                    if self.tree_a[0] == self.start:
                        return path_a[::-1] + path_b
                    else:
                        return path_b[::-1] + path_a
            # swap trees
            self.tree_a, self.tree_b = self.tree_b, self.tree_a
        # no path found
        return None

    def reset(self):
        """Reset the RRT search."""
        self.tree_a = []
        self.tree_b = []

    def dfs(self, end_node):
        """Trace back the path from the end node to the start node."""
        node_ptr = end_node
        path = [node_ptr.q]
        while node_ptr.parent is not None:
            node_ptr = node_ptr.parent
            path.append(node_ptr.q)
        return path
