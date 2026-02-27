from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
import time


class Collector(ABC):
    """
    Abstract class for environment data collection.
    Collects raw episode data from an agent.
    """

    @abstractmethod
    def collect(self, agent):
        """
        Run agent in environment and collect raw info.

        Args:
            agent: agent with act(obs) method, weights already set

        Returns:
            dict of lists, each list has one entry per episode.
            e.g. {'reward': [r1, r2], 'final_x': [x1, x2], ...}
        """
        pass



class BipedalWalkerCollector:
    """
    Collector for BipedalWalker-v3.
    Runs episodes and returns raw per-episode data.

    Args:
        max_steps: max steps per episode
        n_episodes: number of episodes
        seed: random seed
    """

    def __init__(self, max_steps=1600, n_episodes=3, seed=None):
        self.max_steps = max_steps
        self.n_episodes = n_episodes
        self.seed = seed

    def collect(self, agent):
        """
        Run agent and collect raw info.

        Returns:
            dict with keys:
                'reward': list of total reward per episode
                'steps': list of steps survived per episode
                'leg1_contacts': list of arrays of leg1 contact per timestep
                'leg2_contacts': list of arrays of leg2 contact per timestep
                'hull_angles': list of arrays of hull angle per timestep
                'final_x': list of final hull x position per episode
        """
        info = {
            'reward': [],
            'steps': [],
            'leg1_contacts': [],
            'leg2_contacts': [],
            'hull_angles': [],
            'final_x': [],
        }

        for ep in range(self.n_episodes):
            env = gym.make('BipedalWalker-v3')
            seed = self.seed + ep if self.seed is not None else None
            obs, _ = env.reset(seed=seed)
            ep_reward = 0
            leg1 = []
            leg2 = []
            angles = []

            for _ in range(self.max_steps):
                action = agent.act(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                leg1.append(obs[8])
                leg2.append(obs[13])
                angles.append(obs[0])
                if terminated or truncated:
                    break

            info['reward'].append(ep_reward)
            info['steps'].append(len(leg1))
            info['leg1_contacts'].append(np.array(leg1))
            info['leg2_contacts'].append(np.array(leg2))
            info['hull_angles'].append(np.array(angles))
            info['final_x'].append(env.unwrapped.hull.position.x)
            env.close()

        return info



class CartPoleCollector:
    """
    Collector for CartPole-v1.

    Args:
        max_steps: max steps per episode
        n_episodes: number of episodes
        seed: random seed
    """

    def __init__(self, max_steps=500, n_episodes=3, seed=None):
        self.max_steps = max_steps
        self.n_episodes = n_episodes
        self.seed = seed

    def collect(self, agent):
        """
        Returns:
            dict with keys:
                'reward': list of total reward per episode
                'steps': list of steps per episode
                'cart_positions': list of arrays of cart position per timestep
                'actions': list of arrays of actions per timestep
        """
        import gymnasium as gym
        import numpy as np

        info = {
            'reward': [],
            'steps': [],
            'cart_positions': [],
            'actions': [],
        }

        for ep in range(self.n_episodes):
            env = gym.make('CartPole-v1')
            seed = self.seed + ep if self.seed is not None else None
            obs, _ = env.reset(seed=seed)
            ep_reward = 0
            positions = []
            actions = []

            for _ in range(self.max_steps):
                action = agent.act(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                positions.append(obs[0])
                actions.append(action)
                if terminated or truncated:
                    break

            info['reward'].append(ep_reward)
            info['steps'].append(len(positions))
            info['cart_positions'].append(np.array(positions))
            info['actions'].append(np.array(actions))
            env.close()

        return info



class CarRacingCollector:
    """
    Collector for CarRacing-v3 (continuous control).

    Uses a compact handcrafted state from env internals (not pixel observation):
        [speed, angular_velocity, cos(heading_error), sin(heading_error),
         centerline_distance, prev_steer, prev_gas, prev_brake]

    Also logs behavior-relevant metrics for archive/fitness computation.

    Args:
        max_steps: max steps per episode
        n_episodes: number of episodes
        seed: random seed
        lookahead_points: track points ahead used to compute heading error
        speed_scale: normalization scale for speed
        angular_velocity_scale: normalization scale for angular velocity
        centerline_norm: normalization scale for centerline distance
    """

    def __init__(self, max_steps=1000, n_episodes=1, seed=None,
                 lookahead_points=8, speed_scale=40.0,
                 angular_velocity_scale=8.0, centerline_norm=8.0,
                 enable_timing=True):
        self.max_steps = max_steps
        self.n_episodes = n_episodes
        self.seed = seed
        self.lookahead_points = lookahead_points
        self.speed_scale = speed_scale
        self.angular_velocity_scale = angular_velocity_scale
        self.centerline_norm = centerline_norm
        self.enable_timing = enable_timing

        # Aggregated rollout timing stats over all collect() calls.
        self._timing = {
            'collect_calls': 0,
            'total_rollout_steps': 0,
            'total_rollout_time_sec': 0.0,
            'total_rollout_wall_time_sec': 0.0,
            'last_collect_steps': 0,
            'last_collect_time_sec': 0.0,
            'last_collect_wall_time_sec': 0.0,
        }

    def _accumulate_timing(self, rollout_steps, elapsed, wall_time_sec=None):
        wall = float(elapsed) if wall_time_sec is None else float(wall_time_sec)
        self._timing['collect_calls'] += 1
        self._timing['total_rollout_steps'] += int(rollout_steps)
        self._timing['total_rollout_time_sec'] += float(elapsed)
        self._timing['total_rollout_wall_time_sec'] += wall
        self._timing['last_collect_steps'] = int(rollout_steps)
        self._timing['last_collect_time_sec'] = float(elapsed)
        self._timing['last_collect_wall_time_sec'] = wall

    def _wrap_angle(self, angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _extract_track_points(self, env):
        track = getattr(env.unwrapped, "track", None)
        if not track:
            return None

        points = []
        for node in track:
            if len(node) >= 4:
                points.append((float(node[2]), float(node[3])))

        if not points:
            return None
        return np.array(points, dtype=np.float32)

    def _nearest_track_index(self, position_xy, track_points):
        if track_points is None:
            return None
        diffs = track_points - position_xy[None, :]
        d2 = np.sum(diffs ** 2, axis=1)
        return int(np.argmin(d2))

    def _heading_and_centerline(self, env, track_points):
        car = env.unwrapped.car
        position_xy = np.array([car.hull.position.x, car.hull.position.y], dtype=np.float32)

        if track_points is None or len(track_points) < 2:
            return 0.0, 0.0

        nearest_idx = self._nearest_track_index(position_xy, track_points)
        look_idx = (nearest_idx + self.lookahead_points) % len(track_points)
        target = track_points[look_idx]
        target_heading = np.arctan2(target[1] - position_xy[1], target[0] - position_xy[0])
        heading_error = self._wrap_angle(target_heading - float(car.hull.angle))

        centerline_dist = float(np.linalg.norm(track_points[nearest_idx] - position_xy))
        return heading_error, centerline_dist

    def _is_on_track(self, env):
        car = getattr(env.unwrapped, "car", None)
        if car is None or not hasattr(car, "wheels"):
            return True

        has_tiles_attr = False
        for wheel in car.wheels:
            tiles = getattr(wheel, "tiles", None)
            if tiles is None:
                continue
            has_tiles_attr = True
            if len(tiles) > 0:
                return True

        # If wheel-tile contacts are not available in this build, avoid false off-track counts.
        if not has_tiles_attr:
            return True
        return False

    def _build_state(self, env, prev_action, track_points):
        car = env.unwrapped.car
        vx = float(car.hull.linearVelocity.x)
        vy = float(car.hull.linearVelocity.y)
        speed = np.sqrt(vx * vx + vy * vy)
        ang_vel = float(car.hull.angularVelocity)
        heading_error, centerline_dist = self._heading_and_centerline(env, track_points)

        speed_n = np.tanh(speed / self.speed_scale)
        ang_vel_n = np.tanh(ang_vel / self.angular_velocity_scale)
        centerline_n = np.clip(centerline_dist / self.centerline_norm, 0.0, 1.0)

        return np.array([
            speed_n,
            ang_vel_n,
            np.cos(heading_error),
            np.sin(heading_error),
            centerline_n,
            prev_action[0],
            prev_action[1],
            prev_action[2],
        ], dtype=np.float32)

    def collect(self, agent):
        """
        Run agent and collect raw info.

        Returns:
            dict with keys:
                'reward': list of total reward per episode
                'steps': list of steps per episode
                'actions': list of arrays (T, 3)
                'steering': list of arrays (T,)
                'throttle': list of arrays (T,)
                'brake': list of arrays (T,)
                'speeds': list of arrays (T,)
                'lateral_accels': list of arrays (T,)
                'slip_angles': list of arrays (T,)
                'offtrack_steps': list of ints
                'offtrack_ratio': list of floats
                'track_limit_violations': list of ints
                'lap_completion': list of floats in [0, 1]
        """
        t0 = time.perf_counter() if self.enable_timing else None

        info = {
            'reward': [],
            'steps': [],
            'actions': [],
            'steering': [],
            'throttle': [],
            'brake': [],
            'speeds': [],
            'lateral_accels': [],
            'slip_angles': [],
            'offtrack_steps': [],
            'offtrack_ratio': [],
            'track_limit_violations': [],
            'lap_completion': [],
        }

        for ep in range(self.n_episodes):
            env = gym.make('CarRacing-v3', continuous=True)
            seed = self.seed + ep if self.seed is not None else None
            env.reset(seed=seed)

            track_points = self._extract_track_points(env)
            total_track_points = len(track_points) if track_points is not None else 0

            ep_reward = 0.0
            prev_action = np.zeros(3, dtype=np.float32)
            was_on_track = True
            offtrack_steps = 0
            violations = 0

            actions = []
            steering = []
            throttle = []
            brake = []
            speeds = []
            lateral_accels = []
            slip_angles = []

            steps = 0
            for _ in range(self.max_steps):
                state = self._build_state(env, prev_action, track_points)
                action = np.array(agent.act(state), dtype=np.float32).reshape(-1)
                if action.shape[0] != 3:
                    raise ValueError("CarRacingCollector expects agent action shape (3,)")

                action[0] = np.clip(action[0], -1.0, 1.0)  # steer
                action[1] = np.clip(action[1], 0.0, 1.0)   # gas
                action[2] = np.clip(action[2], 0.0, 1.0)   # brake

                _, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                steps += 1

                car = env.unwrapped.car
                vx = float(car.hull.linearVelocity.x)
                vy = float(car.hull.linearVelocity.y)
                speed = np.sqrt(vx * vx + vy * vy)
                ang_vel = float(car.hull.angularVelocity)
                lateral_acc = speed * abs(ang_vel)

                if speed > 1e-8:
                    velocity_heading = np.arctan2(vy, vx)
                    slip = self._wrap_angle(velocity_heading - float(car.hull.angle))
                else:
                    slip = 0.0

                on_track = self._is_on_track(env)
                if not on_track:
                    offtrack_steps += 1
                    if was_on_track:
                        violations += 1
                was_on_track = on_track

                actions.append(action.copy())
                steering.append(float(action[0]))
                throttle.append(float(action[1]))
                brake.append(float(action[2]))
                speeds.append(float(speed))
                lateral_accels.append(float(lateral_acc))
                slip_angles.append(float(slip))

                prev_action = action

                if terminated or truncated:
                    break

            tile_visited = int(getattr(env.unwrapped, "tile_visited_count", 0))
            if total_track_points > 0:
                lap_completion = float(np.clip(tile_visited / total_track_points, 0.0, 1.0))
            else:
                lap_completion = 0.0

            offtrack_ratio = float(offtrack_steps / max(steps, 1))

            info['reward'].append(float(ep_reward))
            info['steps'].append(int(steps))
            info['actions'].append(np.array(actions, dtype=np.float32))
            info['steering'].append(np.array(steering, dtype=np.float32))
            info['throttle'].append(np.array(throttle, dtype=np.float32))
            info['brake'].append(np.array(brake, dtype=np.float32))
            info['speeds'].append(np.array(speeds, dtype=np.float32))
            info['lateral_accels'].append(np.array(lateral_accels, dtype=np.float32))
            info['slip_angles'].append(np.array(slip_angles, dtype=np.float32))
            info['offtrack_steps'].append(int(offtrack_steps))
            info['offtrack_ratio'].append(offtrack_ratio)
            info['track_limit_violations'].append(int(violations))
            info['lap_completion'].append(lap_completion)

            env.close()

        if self.enable_timing:
            elapsed = time.perf_counter() - t0
            rollout_steps = int(np.sum(info['steps']))

            self._accumulate_timing(rollout_steps, elapsed, wall_time_sec=elapsed)

            info['rollout_time_sec'] = float(elapsed)
            info['rollout_steps'] = rollout_steps
            info['steps_per_sec'] = float(rollout_steps / max(elapsed, 1e-12))

        return info

    def record_infos_timing(self, infos, wall_time_sec=None):
        """
        Merge timing from rollout infos (useful when collect() runs in worker processes).
        """
        if not self.enable_timing:
            return
        valid_infos = [info for info in infos if 'rollout_time_sec' in info]
        if not valid_infos:
            return

        # In parallel mode, use the real batch wall-time once (split across infos)
        # to avoid overcounting by summing worker runtimes.
        wall_per_info = None
        if wall_time_sec is not None:
            wall_per_info = float(wall_time_sec) / max(len(valid_infos), 1)

        for info in valid_infos:
            if 'rollout_time_sec' not in info:
                continue
            rollout_steps = int(info.get('rollout_steps', np.sum(info.get('steps', []))))
            elapsed = float(info['rollout_time_sec'])
            self._accumulate_timing(rollout_steps, elapsed, wall_time_sec=wall_per_info)

    def get_timing_stats(self):
        """
        Return aggregated rollout timing stats across all collect() calls.
        """
        total_steps = int(self._timing['total_rollout_steps'])
        total_time = float(self._timing['total_rollout_time_sec'])
        total_wall_time = float(self._timing['total_rollout_wall_time_sec'])
        last_steps = int(self._timing['last_collect_steps'])
        last_time = float(self._timing['last_collect_time_sec'])
        last_wall_time = float(self._timing['last_collect_wall_time_sec'])

        ref_total_time = total_wall_time if total_wall_time > 0 else total_time
        ref_last_time = last_wall_time if last_wall_time > 0 else last_time

        avg_sps = total_steps / ref_total_time if ref_total_time > 0 else 0.0
        last_sps = last_steps / ref_last_time if ref_last_time > 0 else 0.0

        return {
            'collect_calls': int(self._timing['collect_calls']),
            'total_rollout_steps': total_steps,
            'total_rollout_time_sec': total_time,
            'total_rollout_wall_time_sec': total_wall_time,
            'avg_steps_per_sec': float(avg_sps),
            'avg_sec_per_100_steps': float(100.0 / avg_sps) if avg_sps > 0 else None,
            'avg_sec_per_1000_steps': float(1000.0 / avg_sps) if avg_sps > 0 else None,
            'last_collect_steps': last_steps,
            'last_collect_time_sec': last_time,
            'last_collect_wall_time_sec': last_wall_time,
            'last_steps_per_sec': float(last_sps),
        }



class PlanarArmCollector:
    """
    Collector for planar arm inverse kinematics.
    Computes forward kinematics from joint angles.

    Args:
        n_joints: number of joints
        link_length: length of each link (default: 1/n_joints for unit total length)
    """

    def __init__(self, n_joints, link_length=None):
        self.n_joints = n_joints
        self.link_length = link_length if link_length is not None else 1.0 / n_joints

    def collect(self, agent):
        """
        Compute FK from agent's joint angles.

        Returns:
            dict with keys:
                'joint_angles': numpy array of angles
                'end_effector': (x, y) tuple
                'angle_variance': float
        """
        angles = agent.angles
        cumulative = np.cumsum(angles)
        x = np.sum(self.link_length * np.cos(cumulative))
        y = np.sum(self.link_length * np.sin(cumulative))

        return {
            'joint_angles': angles,
            'end_effector': (float(x), float(y)),
            'angle_variance': float(np.var(angles)),
        }
