#!/usr/bin/env python3
"""
Path planning Sample Code with RT-RRT*

uthor: Magnus Knædal
Date: 10.06.2020

"""
import rospy
import math
import os
import sys
import numpy as np
import random
import cProfile
import pstats
import io
import copy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped, Twist
from dynamic_reconfigure.server import Server
from dyn_config.cfg import LocalTuningConfig
from guidance_system.msg import HybridPathSignal

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../RRTStar/")
try:
    # Import utilities
    from _plot_utils import Plot_utils
    from rrt_utils import RRT_utils, prettyfloat
    from grid import Grid
except ImportError:
    raise

class RRTStar(RRT_utils, Plot_utils, Grid):
    """
    Class for informed RRT Star planning
    """

    class Node:        
        """
        Class for representing nodesin the RRT tree.
        """
        def __init__(self, x, y, alpha):            
            self.x        = x
            self.y        = y
            self.parent   = None
            self.children = []
            self.alpha    = alpha
            self.kappa    = 0.0
            self.d        = float('Inf')
            self.cost     = 0.0

        def __repr__(self):
            """Print format of node

            Returns:
                [type] -- [description]
            """            
            return ('{0}, {1}, {2}, {3} \n'.format(self.x, self.y, np.rad2deg(self.alpha), self.cost))


    def __init__(self, obstacle_list, goal_region,
                 path_resolution, goal_sample_rate, beta,
                 max_alpha, max_kappa, k_max, K, r_min, r_max, grid_dim, cost_function, 
                 delta, epsilon, zeta, occupied_thres, rviz_tuning_plt):
        """
        Init RT-RRT*.
        :param:
            start: [x, y, theta]
            goal: [x, y, theta]
            obstacle_list: List of obstacles [x, y, radius]
            goal_region: Radius of goal region. If nodes inside, goal is found
            expand_dis: Expanding distance
            path_resolution: Controls the roughness of the steering
            goal_sample_rate: percentage
            beta: Deviding sampling between uniform and ellipse
            max_alpha: Maximum angle between two consecutive nodes
            max_kappa: Maximum "curvature"
            k_max: Maximum number of neighbours  around a node.
            K: Maximum steps for planning a path if goal not found
            r_s: Minimum allowed distance between the nodes in the tree
            grid_dim: Dimensions of grid, number of cells (n times n)
            cost_function: Which cost function to use. Either "distance", "curvature", "obstacle", or "total"
            delta: Distance cost function tuning paramereter
            epsilon:  Obstacle cost function tuning paramereter
            zeta: Curvature cost function tuning paramereter
            occupied_thres: Threshold for grid cell to be treated occupied
            rviz_tuning_plt: Tuning size of linewidth (on tree etc.) of plotting in rviz.
        """

        ### Ros node, pubs and subs ### 
        rospy.init_node('RRT', anonymous=True)
        self.pub_nodes         = rospy.Publisher('nav_syst/local_planner/plotting/nodes', Marker, queue_size = 1)
        self.pub_edges         = rospy.Publisher('nav_syst/local_planner/plotting/edges', MarkerArray, queue_size = 5)
        self.pub_blocked_nodes = rospy.Publisher('nav_syst/local_planner/plotting/blocked_nodes', Marker, queue_size = 1)
        self.pub_blocked_edges = rospy.Publisher('nav_syst/local_planner/plotting/blocked_edges', MarkerArray, queue_size = 1)
        self.pub_path          = rospy.Publisher('nav_syst/local_planner/plotting/path', Marker, queue_size = 1)
        self.pub_start_goal    = rospy.Publisher('nav_syst/local_planner/plotting/start_goal', Marker, queue_size = 1)
        self.pub_root          = rospy.Publisher('nav_syst/local_planner/plotting/root', Marker, queue_size = 1)
        self.pub_ellipse       = rospy.Publisher('nav_syst/local_planner/plotting/ellipse', Marker, queue_size = 1)
        self.pub_obst          = rospy.Publisher('nav_syst/local_planner/plotting/obstacles', MarkerArray, queue_size = 1)
        self.pub_goal_region   = rospy.Publisher('nav_syst/local_planner/plotting/goal_region', Marker, queue_size = 1)

        self.pub_local_path = rospy.Publisher('nav_syst/local_planner/path', Path, queue_size = 20)
        rospy.Subscriber('/observer/eta/ned', Twist, self.eta_listener)
        rospy.Subscriber('GuidanceSystem/HybridPathSignal', HybridPathSignal, self.hybrid_signal_listener)

        self.eta = np.zeros((3,1))
        self.eta_d = np.zeros((3,1))
        self.global_wps = []
        rospy.Subscriber("/nav_syst/global/path", Path, self.wp_listener)
        # Wait for global wps
        while len(self.global_wps) == 0:
            continue

        ### Parameters ###
        self.root      = self.Node(self.global_wps[0][0], self.global_wps[0][1], self.global_wps[0][2])
        self.start     = self.Node(self.global_wps[0][0], self.global_wps[0][1], self.global_wps[0][2])
        self.goal_node = self.Node(self.global_wps[1][0], self.global_wps[1][1], self.global_wps[1][2])
        
        self.goal_region      = goal_region
        self.path_resolution  = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.found_goal       = False
        self.target           = None
        self.obstacle_list    = obstacle_list
        self.beta             = beta        
        self.max_alpha        = max_alpha
        self.max_kappa        = max_kappa
        self.k_max            = k_max
        self.K                = K # Maximum steps for planning a path if goal not found
        self.r_min            = r_min # Controls the closeness of the nodes
        self.r_max            = r_max # Controls the closeness of the nodes
        self.cost_function    = cost_function
        self.delta            = delta
        self.epsilon          = epsilon
        self.zeta             = zeta
        self.occupied_thres   = occupied_thres
        self.rviz_tuning_plt  = rviz_tuning_plt

        self.node_list      = [self.root] # Nodes in RRT, representing the tree
        self.visited_set    = [] # Keep track of "visited" nodes in path-planner to not get traped in local minima
        self.Q_r            = [] # Queue for random rewiring
        self.Q_s            = [] # Queue for rewiring of root
        self.near_goal_list = [] # Lists of nodes near goal

        rospy.Subscriber("/costmap_node/costmap/costmap", OccupancyGrid, self.callback_map)
        ### Init map ###
        self.map = None
        while self.map is None:
            continue

        self.grid = np.reshape(np.array(self.map.data), (self.map.info.width, self.map.info.height) )
        self.index_free_space_list = []
        self.get_free_space(self.map, self.grid)
        self.search_space_area = self.map.info.resolution**2 * len(self.index_free_space_list)
        self.gridmap = Grid(self.map.info.origin.position.x, self.map.info.origin.position.x + self.map.info.resolution*self.map.info.width, 
        self.map.info.origin.position.y, self.map.info.origin.position.y + self.map.info.resolution*self.map.info.height, grid_dim)
        self.gridmap.add_index_to_cell(self.root.x, self.root.y, self.node_list.index(self.root))

        srv = Server(LocalTuningConfig, self.parameter_callback)
    
    def update_local_planner(self, current_wp, next_wp):
        """Update planner when new global waypoint

        Arguments:
            current_wp {list} -- [x,y,psi]
            next_wp {list} -- [x,y,psi]
        """

        self.start          = self.Node(current_wp[0], current_wp[1], current_wp[2])
        self.goal_node      = self.Node(next_wp[0], next_wp[1], next_wp[2])

        self.visited_set    = []
        self.near_goal_list = [ node for node in self.node_list if self.is_near_goal(node, self.goal_node)]
        if len(self.near_goal_list) != 0:
            self.found_goal = True
            self.target     = self.get_best_target_node(self.goal_node)
        else:
            self.found_goal = False
            self.target = None

    def realtime_planner(self, global_wps):
        """The RT-RRT* planner, main loop.

        Keyword Arguments:
            animation {bool} -- [if animate or not] (default: {True})

        Returns:
            [type] -- [final path]
        """

        iter = 1
        for n in range(1 ,len(global_wps)):
            
            if n != 1:
                self.update_local_planner(global_wps[n-1], global_wps[n])

            current_path   = []
            cBest          = float('Inf')

            while True:
                print("Iteration %i", iter)
                iter += 1

                # Update obstacle
                if self.obstacle_list[0][0] > self.map.info.origin.position.x + 4:
                    self.obstacle_list[0][0] -= 0.5

                if self.obstacle_list[1][0] > self.map.info.origin.position.x + 4: 
                    self.obstacle_list[1][0] -= 0.5

                # Update root, goal, obstacles etc.  
                self.block_obstacle_branches(self.obstacle_list)

                # New root
                if len(current_path) != 0:
                    self.visited_set = []
                    self.Q_s         = []
                    self.root = current_path[0]
                    self.root.parent.children.remove(self.root)
                    
                    # Make children from previous root get rewired for sure
                    if len(self.root.parent.children) > 0:
                        self.root.parent.cost = 10000
                        self.propagate_cost_to_leaves(self.root.parent)

                    self.root.parent = None
                    self.root.cost = 0 # reset cost

                if (self.goal_node.x - self.eta_d[0, 0])**2 + (self.goal_node.y - self.eta_d[1, 0])**2 < self.goal_region**2:
                    break

                # Expand and rewire
                if len(current_path) == 0:
                    for _ in range(150):
                        self.expand_and_rewire(self.root, self.goal_node, cBest)
                else:                
                    while math.sqrt( (current_path[0].x - self.eta_d[0, 0])**2 + (current_path[0].y - self.eta_d[1, 0])**2 ) > 1.6:
                        self.expand_and_rewire(self.root, self.goal_node, cBest)

                current_path, cBest = self.plan_path(self.root, self.goal_node, self.K, self.visited_set)
                
                if len(current_path) != 0:
                    self.publish_local_wps(self.root, current_path[0])

                current_path.insert(0, self.root) # Current_path do not contain root. Add for plotting.
                self.draw_graph(self.root, self.goal_node, cBest, current_path)
                self.py_plotting(self.node_list, self.root, self.goal_node, self.goal_region, current_path, self.obstacle_list, "", iter, cBest)
                current_path.pop(0)

            print("At new wp %i" % n)

    def publish_local_wps(self, root, next):
        """Publish current and next WP to guidance system
        # NOTE: send by radians through z-field of pose msg. Do not use quaternions part of path msg.

        Arguments:
            root {node} -- [current node]
            next {node} -- [next node]
        """        
        wp_list = Path()

        wp = PoseStamped()
        wp.pose.position.x = root.x
        wp.pose.position.y = root.y
        wp.pose.position.z = root.alpha
        wp_list.poses.append(wp)

        wp = PoseStamped()
        wp.pose.position.x = next.x
        wp.pose.position.y = next.y
        wp.pose.position.z = next.alpha
        wp_list.poses.append(wp)

        self.pub_local_path.publish(wp_list)        

    def expand_and_rewire(self, root, goal, cBest):
        """Tree expansion and rewiring of RT-RRT*.

        Arguments:
            root {[node]} -- [tree root node]
            goal {[node]} -- [goal node]
            cBest {[double]} -- [current best cost. For informed sample.]
        """

        rnd_node = self.informed_sample(root, goal, cBest)
        index = self.gridmap.get_nearest_index_X_si(rnd_node, self.node_list)
        while index == -1: # no one close
            rnd_node = self.informed_sample(root, goal, cBest)
            index = self.gridmap.get_nearest_index_X_si(rnd_node, self.node_list)

        nearest_node = self.node_list[ self.gridmap.get_nearest_index_X_si(rnd_node, self.node_list) ]
        new_node = self.steer(nearest_node, rnd_node, self.r_max)

        if self.check_obstacle_collision(new_node, nearest_node, self.obstacle_list):
            near_inds = self.find_nodes_near(new_node)
            d = self.euclidian_distance(nearest_node, new_node)

            if len(near_inds) < self.k_max and len(near_inds) > 0 and d > self.r_min:
                # Returns none if no feasible. 
                new_node = self.choose_parent(new_node, near_inds)
                if new_node:
                    self.node_list.append(new_node)
                    self.gridmap.add_index_to_cell(new_node.x, new_node.y, len(self.node_list)-1)
                    self.Q_r.append(new_node)

                    # Check if node is near goal, then check if constraints are ok
                    if self.is_near_goal(new_node, goal):
                        if self.check_constraints(new_node, goal) \
                        and self.check_obstacle_collision(new_node, goal, self.obstacle_list) \
                        and self.check_wall_collision(new_node, goal):
                            
                            self.found_goal = True
                            self.near_goal_list.append(new_node)
                            self.target = self.get_best_target_node(goal)
            
            else:
                self.Q_r.append(nearest_node)
 
            self.rewire_random_nodes()

        self.rewire_from_root(root)

    def rewire_random_nodes(self):
        """This function checks if the cost to the nodes in near_inds is less through new_node as compared to their older costs, 
        then its parent is changed to new_node.
        """        
        
        
        startTime = rospy.Time.now()
        duration = rospy.Duration(0.05)

        # Repeat until time is up or Qr is empty
        while (len(self.Q_r) > 0) and (rospy.Time.now() < startTime+duration):
            node = self.Q_r.pop()
            near_inds = self.find_nodes_near(node)
            
            for i in near_inds:
                near_node = self.node_list[i]
                new_cost = self.calc_new_cost(node, near_node)
                improved_cost = new_cost < near_node.cost 

                if improved_cost \
                and self.check_constraints(node, near_node) \
                and self.check_obstacle_collision(node, near_node, self.obstacle_list) \
                and self.check_wall_collision(node, near_node):

                    if near_node.parent != None: 
                        near_node.parent.children.remove(near_node)
                    near_node.parent = node
                    node.children.append(near_node)
                    self.update_node_values(near_node)

                    near_node.cost = new_cost
                    self.propagate_cost_to_leaves(near_node)

                    self.unblock_parents(near_node, self.visited_set)
                    # Append edge_node to Q_r
                    self.Q_r.append(near_node)

    def rewire_from_root(self, root):
        """This function checks if the cost to the nodes in near_inds is less through root as compared to their older costs, 
        then its parent is changed to new_node.

        Arguments:
            root {[node]} -- [tree root]
        """        

        if len(self.Q_s) == 0:
            self.Q_s.append(root)
        
        # Keep track of nodes allready added to Q_s
        Q_s_old = []

        startTime = rospy.Time.now()
        duration = rospy.Duration(0.05)

        while len(self.Q_s) > 0 and (rospy.Time.now() < startTime+duration):
            node = self.Q_s.pop()
            Q_s_old.append(node)
            near_inds = self.find_nodes_near(node)

            for i in near_inds:
                near_node         = self.node_list[i]
                new_cost          = self.calc_new_cost(node, near_node)
                improved_cost     = new_cost < near_node.cost 

                if improved_cost \
                and self.check_constraints(node, near_node) \
                and self.check_obstacle_collision(node, near_node, self.obstacle_list) \
                and self.check_wall_collision(node, near_node):
                    
                    if near_node.parent != None: 
                        near_node.parent.children.remove(near_node) # remove from old parents child list
                    near_node.parent = node # set new parent
                    node.children.append(near_node) # set new child for parent
                    self.update_node_values(near_node)

                    near_node.cost = new_cost
                    self.propagate_cost_to_leaves(near_node)

                    self.unblock_parents(near_node, self.visited_set)

                    # Append edge_node to Q_s if not been added before
                    if near_node not in Q_s_old:
                        self.Q_s.append(near_node)

    def plan_path(self, root, goal, K, visited_set):
        """Plans next path. If goal is found we generate the path to goal,
        else we plan a path for K steps.

        Arguments:
            root {[type]} -- [description]
            goal {[type]} -- [description]

        Returns: the path and the current best cost, cBest (Inf if goal not found). If 
        Note: root not added to path.
            [path] -- [list of nodes]
        """

        # If goal is found, generate path
        if self.found_goal == True:
            path = self.generate_final_path(self.target)

        if self.found_goal == True:
            # Check if path found do not lead to root or final path is blocked by obstacles
            if path[0] not in root.children or path[0].cost == float('Inf'):
                # Then we plan K step path
                if len(root.children) == 0:
                    print("Root has no children, then stay in root1")
                    return [], float('Inf') # Stay in root

                else:
                    return self.plan_K_step_path(root, goal, K, visited_set)
            else:
                # Calculate cBest
                path.insert(0, root)
                cBest = self.get_path_len(path)
                path.pop(0)

                print("Path to goal found")

                return path, cBest 
            
        # If root has no children, then stay in root
        if len(root.children) == 0:
            print("Root has no children, then stay in root")
            return [], float('Inf') # Stay in root

        else:
            return self.plan_K_step_path(root, goal, K, visited_set)

    def plan_K_step_path(self, root, goal, K, visited_set):
        """Plans a k-step path if possible. If the place does not bring us to a better place, we stay in root.
        Essentially performs a greedy search.

        Arguments:
            root {[node]} -- [root]
            goal {[node]} -- [goal]
            K {[int]} -- [How many steps]
            visited_set {[list]} -- [set of visitied nodes. Ensure to not get stuck in local minima.]

        Returns:
            [type] -- [path, and cost of path cBest.]
        """
        #new_k_step_path = False
        node_i = self.get_minimum_cost_child(root, goal, visited_set)

        for i in range(1, K):
            if i == K-1 or len(node_i.children) == 0 or self.check_if_all_children_blocked(node_i, visited_set):
                path = self.generate_K_step_path(node_i, root)
                visited_set.append(node_i)
                #new_k_step_path = True
                break
            node_i = self.get_minimum_cost_child(node_i, goal, visited_set)

        # Choose to stay in root or follow the path.
        # If it leads us closer to goal, we choose to go
        root_heurstic = self.calc_heuristic(root, goal)
        k_root_heuristic = self.calc_heuristic(path[-1], goal)
        if k_root_heuristic <= root_heurstic:
            print("K-step")
            return path, float('Inf') # Choose new path.
        else:
            print("Stay in root")
            return [], float('Inf') # Stay in root        

    def find_nodes_near(self, node):
        """Finds nearest nodes in X_si.

        Arguments:
            node {[node]} -- [node]

        Returns:
            [list] -- [list of nodes]
        """        
        nnode = len(self.node_list) + 1
        epsilon = math.sqrt( (self.search_space_area * self.k_max) / (math.pi * nnode))

        if epsilon < self.r_max:
            r_max = self.r_max
        else:
            r_max = epsilon
        
        # First find nodes nearby
        X_si = self.gridmap.get_X_si_indices(node.x, node.y)

        # Avoding square root by checking the square instead
        dist_list = [(self.node_list[i].x - node.x) ** 2 +
                     (self.node_list[i].y - node.y) ** 2 for i in X_si]

        near_inds = [X_si[i] for dist, i in zip(dist_list, range(0, len(dist_list))) if dist <= self.r_max**2]

        return near_inds

    def informed_sample(self, root, goal, cMax):
        """Performe a informed sample. Returns independent and identically distributed (i.i.d.) samples from the state space.
        
        Arguments:
            root {[node]} -- [tree root]
            goal {[node]} -- [goal]
            cMax {[double]} -- [param for informed sample calculation]

        Returns:
            [node] -- [a sampled node]
        """        


        Pr = random.uniform(0, 1)

            # Sample line between nearest node to goal and goal
        if Pr > (1-self.goal_sample_rate):
            a = (goal.y - root.y) / (goal.x - root.x)
            b = goal.y - a * goal.x

            # Sample from a free grid cell.
            cell = -1
            while True:
                x_sample = random.uniform(root.x, goal.x)
                y_sample = a * x_sample + b
                cell = self.get_grid_cell(x_sample, y_sample)
                if cell != -1 and cell <= self.occupied_thres:
                    sample_node = self.Node(x_sample, y_sample, 0)
                    break
            
            return sample_node

        # Sample unfiform
        if ( Pr <= (1-self.goal_sample_rate)/self.beta ) or (cMax == float('Inf')):
            pos = self.get_ned_position(random.choice(self.index_free_space_list))
            return self.Node(pos[0], pos[1], 0)
        
        # Sample ellipse
        else:                        
            cMin, xCenter, a1, _ = self.compute_sampling_space(root, goal)
            C = self.rotation_to_world_frame(a1)

            r = [cMax / 2.0,
                 math.sqrt(cMax ** 2 - cMin ** 2) / 2.0,
                 math.sqrt(cMax ** 2 - cMin ** 2) / 2.0]
            L = np.diag(r)

            # Sample from a free grid cell.
            cell = -1
            while True:
                xBall = self.sample_unit_ball()
                rnd = np.dot(np.dot(C, L), xBall) + xCenter
                cell = self.get_grid_cell(rnd[0][0], rnd[1][0])
                # if cell is not occupied and inside dimensions
                if cell != None and cell != -1 and cell <= self.occupied_thres:
                    sample_node = self.Node(rnd[0][0], rnd[1][0], 0)
                    break
            return sample_node

    def is_near_goal(self, node, goal):
        """Given a pose, the function is_near_goal returns True if and only if the state is in the goal region, as defined.

        Arguments:
            node {[type]} -- [description]
            goal {[type]} -- [description]

        Returns:
            [bool] -- [true if nodeis close to goal]
        """        
        d = self.euclidian_distance(node, goal)
        if d < self.goal_region:
            return True
        return False

    def choose_parent(self, new_node, near_inds):
        """Set parent of new node to the one found with lowest cost and satisfying constraints.

        Arguments:
            new_node {[type]} -- [description]
            filtered_inds {[list]} -- [list of possible indices to be parent]

        Returns:
            [node] -- [ the node] # TODO: not necassery.
        """
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]

            # If node obstacle and wall collision
            if self.check_constraints(near_node, new_node) \
            and self.check_obstacle_collision(near_node, new_node, self.obstacle_list) \
            and self.check_wall_collision(near_node, new_node):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float('Inf'))  # the cost of collision node
        
        min_cost = min(costs)
        if min_cost == float('Inf'):
            return None

        # Set parent to the one found with lowest cost
        min_ind = near_inds[costs.index(min_cost)]
        parent_node = self.node_list[min_ind]

        parent_node.children.append(new_node)
        new_node.parent = parent_node
        new_node.cost = min_cost

        self.update_node_values(new_node)
        self.unblock_parents(new_node, self.visited_set)
    
        return new_node

    def update_node_values(self, node):
        """Updates values for new or rewired node.

        Arguments:
            node {[type]} -- [description]
        """        
        d, alpha = self.euclidian_distance_and_angle(node.parent, node)
        node.d = d
        node.alpha = alpha

        if d == 0 or node.parent.d == 0 or abs(self.ssa(node.parent.alpha - node.alpha)) > self.max_alpha:
            node.kappa = float('Inf')
        else:
            node.kappa = (2*math.tan(abs(self.ssa(node.parent.alpha - node.alpha)))) / min(node.parent.d, node.d)
        
    def get_best_target_node(self, goal):
        """
        Finds best target in near_goal_list

        Arguments:
            goal {[type]} -- [description]

        Returns:
            [node] -- [best target node close to goal]
        """        
        """
        """
        best_target = self.near_goal_list[0]

        for node in self.near_goal_list:
            if node.cost < best_target.cost:
                best_target = node

        return best_target

    def find_nodes_near_obstacle(self, obstacle):
        """Finds nodes near to a given obstacle

        Arguments:
            obstacle {[list]} -- [list of obstacles with x,y,radius]

        Returns:
            [list] -- [list of indicies close to obstacle]
        """        
        """
        obstacle = [x, ,y, radius]
        returns nearest nodes in X_si.
        """
        obst_x = obstacle[0]
        obst_y = obstacle[1]
        obst_r = obstacle[2]

        # First find nodes nearby
        X_si = self.gridmap.get_X_si_indices(obst_x, obst_y)
        dist_list = [(self.node_list[i].x - obst_x) ** 2 +
                     (self.node_list[i].y - obst_y) ** 2 for i in X_si]
        near_inds = [X_si[i] for dist, i in zip(dist_list, range(0, len(dist_list))) if dist <= obst_r ** 2]

        return near_inds
    
    def block_obstacle_branches(self, obstacles):
        """Block branches intersecting with obstacles. Note that the grid size must be big enough
        to capture the obstacle region in order to ensure that all nodes are blocked.

        Arguments:
            obstacles {[list]} -- [list of obstacles x,y,radius]
        """        
        for obstacle in obstacles:
            near_inds = self.find_nodes_near_obstacle(obstacle)
            for index in near_inds:
                node = self.node_list[index]
                if node != self.root:
                    node.cost = float('Inf')
                    self.propagate_cost_to_leaves(node)
        
    def check_constraints(self, from_node, to_node):
        """ Checks constraints related to angle and cruvature.

        Arguments:
            from_node {[type]} -- [description]
            to_node {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        to_node_d, to_node_alpha = self.euclidian_distance_and_angle(from_node, to_node)

        if to_node_d == 0 or from_node.d == 0:
            kappa_next = float('Inf')
        else:
            kappa_next = (2*math.tan(abs(self.ssa(from_node.alpha - to_node_alpha)))) / min(from_node.d, to_node_d)

        alpha_ok = abs( self.ssa(from_node.alpha - to_node_alpha)) < self.max_alpha        
        kappa_ok = kappa_next < self.max_kappa

        if kappa_ok and alpha_ok:
            return True # Constraints ok
        else:
            return False # Not ok

    def propagate_cost_to_leaves(self, parent_node):
        """ Propagtes the cost of parent all the way down to the leaf nodes.  

        Arguments:
            parent_node {[type]} -- [description]
        """
        if len(parent_node.children) > 0:
            for child in parent_node.children:
                if child.parent == parent_node:
                    child.cost = self.calc_new_cost(parent_node, child)
                    self.propagate_cost_to_leaves(child)

    def calc_new_cost(self, from_node, to_node):
        """Calculate cost functions.
        :param:
            c_d - distance cost
            c_o - obstacle cost
            c_c - curvature cost
            delta - c_d tuning
            epsilon - c_o tuning
            zeta - c_c tuning

        Arguments:
            from_node {[type]} -- [description]
            to_node {[type]} -- [description]

        Returns:
            [double] -- [cost]
        """

        cost = 0

        # Distance cost
        if self.cost_function == "distance" or self.cost_function == "total":
            cost += from_node.cost + self.euclidian_distance(from_node, to_node) * self.delta

        # Obstacle cost
        elif self.cost_function == "obstacle" or self.cost_function == "total":
            if len(self.obstacle_list) > 0:
                cost += from_node.cost + 0.5 * self.epsilon * (1 / self.get_min_obstacle_distance(to_node, self.obstacle_list))**2
            else:
                cost += 0
        
        # Curvature cost
        elif self.cost_function == "curvature" or self.cost_function == "total":
            d, alpha_next = self.euclidian_distance_and_angle(from_node, to_node)
            if d == 0 or from_node.d == 0 or abs(self.ssa(from_node.alpha - alpha_next)) > self.max_alpha:
                cost += float('Inf')
            else:
                RRTStar.get_sum_c_c.counter = 0
                kappa_next = (2*math.tan(abs(self.ssa(from_node.alpha - alpha_next)))) / min(from_node.d, d)

                cost += (( max(self.get_max_kappa(from_node), kappa_next) 
                        + (self.get_sum_c_c(from_node) + kappa_next) / (RRTStar.get_sum_c_c.counter) )) * self.zeta
        
        return cost

    def steer(self, from_node, to_node, extend_length = float('Inf')):
        """Given two nodes n1, n2, the function returns a new node n3 such that n3 is closer to n2 than n1 is.
        n3 will minimze the distance n3-n2, but at the same time maintain that the distance n1-n3 <= extend_length.

        Arguments:
            from_node {[type]} -- [description]
            to_node {[type]} -- [description]

        Keyword Arguments:
            extend_length {[type]} -- [description] (default: {float('Inf')})

        Returns:
            [type] -- [description]
        """

        new_node = self.Node(from_node.x, from_node.y, 0)
        d, theta = self.euclidian_distance_and_angle(new_node, to_node)
        if d < extend_length:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)

        new_node.parent = from_node

        return new_node

    ### ROS callbacks ###

    def callback_map(self, occupancy_grid_msg):
        self.map = occupancy_grid_msg

    def wp_listener(self, path):
        """
        Waypoint-listener
        """
        global_wps = np.zeros((len(path.poses), 3))

        for i in range(len(path.poses)):
            wp = [path.poses[i].pose.position.x, path.poses[i].pose.position.y, path.poses[i].pose.position.z]
            
            global_wps[i] = wp

        # If new waypoints. Else do nothing.
        if not np.array_equal(global_wps, self.global_wps):
            self.global_wps = global_wps

    def parameter_callback(self, config, level):
        """ Callback function for updating parameters.

        Parameters
        ----------
        config : ParameterGenerator()
            configuration parameters

        """

        execute = config.execute_local_planner
        if execute == True:
            self.realtime_planner(self.global_wps)

        return config

    def eta_listener(self, eta):
        """
        Listens to eta (NED) position.
        """
        # [x, y, psi]

        deg2rad = math.pi/180
        self.eta = np.array([ [eta.linear.x], [eta.linear.y], [deg2rad*eta.angular.z] ])

    def hybrid_signal_listener(self, signal):
        self.eta_d = np.array([[signal.eta_d.x], [signal.eta_d.y], [signal.eta_d.theta] ])


def main():
    # [x, y, radius]
    obstacleList = [
        [177, 1149, 2],
        [165, 1129, 2]
        ]

    # Set Initial parameters
    rrt_star = RRTStar(obstacle_list    = obstacleList,          # List of obstacles [x, y, radius]
                       goal_region      = 4,                     # Radius of goal region. If nodes inside, goal is found
                       path_resolution  = 0.1,                   # Controls the roughness of the steering
                       goal_sample_rate = 0.05,                   # percentage
                       beta             = 10,                    # deviding sampling between uniform and ellipse
                       max_alpha        = math.pi/2,             # maximum angle between two consecutive nodes
                       max_kappa        = 2,                    # maximum "curvature"
                       k_max            = 100,                    # Maximum number of neighbours around a node.
                       K                = 5,                     # Maximum steps for planning a path if goal not found
                       r_min            = 1.5,                     # Minimum allowed distance between the nodes in the tree
                       r_max            = 6,                     # Maximum allowed distance between the nodes in the tree
                       grid_dim         = 8,                     # number of cells
                       cost_function    = "distance",            # Either "distance", "curvature", "obstacle", or "total"
                       delta            = 1,                     # Distance cost function tuning paramereter
                       epsilon          = 1,                     # Obstacle cost function tuning paramereter
                       zeta             = 1,                     # Curvature cost function tuning paramereter
                       occupied_thres   = 20,                   # Threshold for grid cell to be treated occupied
                       rviz_tuning_plt  = 0.1                    # Tuning size of linewidth (on tree etc.) of plotting in rviz.
                                        )
    
if __name__ == '__main__':
    # Profiling
    #cProfile.run('main()', 'rrt_profiling.txt')
    #p = pstats.Stats('rrt_profiling.txt')
    #p.sort_stats('cumulative').print_stats(20)

    main()

    rospy.spin()


