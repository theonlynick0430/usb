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
        """Return a random configuration from the joint range."""
        return np.random.uniform(*self.jnt_range)
    
    def is_valid(self, q):
        """Return if a configuration is valid."""
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
    
    def plan(self, q_start, q_goal, smooth=True):
        """
        Plan a path from q_start to q_goal.

        Args:
            q_start: start configuration
            q_goal: goal configuration
            smooth: whether to smooth the path
        """
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
                    path = None
                    if self.tree_a[0] == self.start:
                        path = path_a[::-1] + path_b
                    else:
                        path = path_b[::-1] + path_a
                    if smooth:
                        path = self.smooth(path)
                    return path
            # swap trees
            self.tree_a, self.tree_b = self.tree_b, self.tree_a
        # no path found
        return None
    
    def smooth(self, path, max_attempts=100):
        """
        Smooth the given path by shortcutting collision-free straight segments,
        then re-interpolate the final path to ensure step size resolution.

        Args:
            path: list of joint configurations (np.ndarray)
            max_attempts: number of shortcut attempts

        Returns:
            Smoothed and interpolated path.
        """
        path = path.copy()
        for _ in range(max_attempts):
            if len(path) < 3:
                break

            i, j = sorted(np.random.choice(len(path), size=2, replace=False))
            if j - i < 2:
                continue

            q_i, q_j = path[i], path[j]

            # attempt shortcut
            num_steps = int(np.ceil(np.linalg.norm(q_j - q_i) / self.step_size))
            success = True
            for alpha in np.linspace(0, 1, num_steps):
                q_interp = (1 - alpha) * q_i + alpha * q_j
                if not self.is_valid(q_interp):
                    success = False
                    break

            if success:
                # replace segment with direct connection
                path = path[:i+1] + path[j:]

        # ensure step size resolution
        smoothed = [path[0]]
        for i in range(1, len(path)):
            q_start, q_end = path[i - 1], path[i]
            delta = q_end - q_start
            dist = np.linalg.norm(delta)
            steps = max(1, int(np.ceil(dist / self.step_size)))
            for alpha in np.linspace(0, 1, steps, endpoint=False)[1:]:
                smoothed.append(q_start + alpha * delta)
            smoothed.append(q_end)

        return smoothed

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
