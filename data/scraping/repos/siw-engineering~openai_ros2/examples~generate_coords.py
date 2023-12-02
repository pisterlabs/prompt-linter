from openai_ros2.tasks import LobotArmRandomGoal
import rclpy

rclpy.init()
node = rclpy.create_node("SomeName")
robot = "Dummy"
task_kwargs = {
                'accepted_dist_to_bounds': 0.001,
                'accepted_error': 0.001,
                'reach_target_bonus_reward': 0.0,
                'reach_bounds_penalty': 0.0,
                'contact_penalty': 0.0,
                'episodes_per_goal': 1,
                'goal_from_buffer_prob': 0.0,
                'num_adjacent_goals': 0,
                'random_goal_seed': 10,
                'is_validation': False,
                'normalise_reward': True
            }
task = LobotArmRandomGoal(node, robot, **task_kwargs)
target_coords = []
for _ in range(1000):
    target_coords.append(task.target_coords)
    task.reset()

print(target_coords)