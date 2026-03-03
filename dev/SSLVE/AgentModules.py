import numpy as np


class MLP_Agent:
    """
    MLP agent that takes external flat weight vector.
    ReLU hidden activations, configurable output activation.

    Args:
        architecture: list of layer dims, e.g. [4, 32, 2]
        output_activation:
            - 'argmax' (discrete)
            - 'tanh' (continuous)
            - 'car_racing' ([steer, accel, _] -> [tanh(steer), relu(accel), relu(-accel)])
    """

    def __init__(self, architecture, output_activation='argmax'):
        self.architecture = architecture
        self.output_activation = output_activation
        self.weights = []
        self.biases = []

    def get_weight_dim(self):
        """Total number of parameters in flat weight vector."""
        dim = 0
        for i in range(len(self.architecture) - 1):
            dim += self.architecture[i] * self.architecture[i + 1]  # weight matrix
            dim += self.architecture[i + 1]  # bias
        return dim

    def set_weights(self, flat_weights):
        """
        Unflatten weight vector into layer weights and biases.

        Args:
            flat_weights: 1D numpy array of length get_weight_dim()
        """
        self.weights = []
        self.biases = []
        offset = 0
        for i in range(len(self.architecture) - 1):
            fan_in = self.architecture[i]
            fan_out = self.architecture[i + 1]
            w_size = fan_in * fan_out
            W = flat_weights[offset:offset + w_size].reshape(fan_in, fan_out)
            offset += w_size
            b = flat_weights[offset:offset + fan_out]
            offset += fan_out
            self.weights.append(W)
            self.biases.append(b)

    def act(self, obs):
        """
        Forward pass. ReLU for hidden layers, output_activation for output.

        Args:
            obs: 1D numpy array (observation)

        Returns:
            int action (argmax) or numpy array
        """
        x = np.array(obs, dtype=np.float64)
        for i in range(len(self.weights) - 1):
            x = x @ self.weights[i] + self.biases[i]
            x = np.maximum(x, 0)
        x = x @ self.weights[-1] + self.biases[-1]

        if self.output_activation == 'argmax':
            return int(np.argmax(x))
        elif self.output_activation == 'tanh':
            return np.tanh(x)
        elif self.output_activation == 'car_racing':
            if x.shape[0] != 3:
                raise ValueError("output_activation='car_racing' requires output dim = 3")
            steer = np.tanh(x[0])
            # Use a single acceleration channel to avoid persistent gas+brake conflicts.
            accel = np.tanh(x[1])
            gas = max(accel, 0.0)
            brake = max(-accel, 0.0)
            return np.array([steer, gas, brake], dtype=np.float32)
        else:
            return x



class PlanarArmAgent:
    """
    Agent for planar arm IK. Stores joint angles as weights.

    Args:
        architecture: int, number of joints (n_joints)
    """

    def __init__(self, architecture, **kwargs):
        self.n_joints = architecture
        self.angles = None

    def get_weight_dim(self):
        return self.n_joints

    def set_weights(self, flat_weights):
        self.angles = flat_weights



class EvoGymAgent:
    """
    Agent for EvoGym morphology evolution.
    Genome is a continuous vector that maps to a 2D voxel grid.
    Uses open-loop sinusoidal actuation (no neural network).

    Args:
        architecture: tuple (rows, cols), e.g. (5, 5)
        freq: sinusoidal controller frequency in Hz
    """

    # Material type constants
    EMPTY = 0
    RIGID = 1
    SOFT = 2
    H_ACT = 3
    V_ACT = 4

    def __init__(self, architecture, **kwargs):
        self.grid_shape = architecture
        self.n_voxels = architecture[0] * architecture[1]
        self.raw_genome = None
        self.freq = kwargs.get('freq', 2.0)

    def get_weight_dim(self):
        return self.n_voxels

    def set_weights(self, flat_weights):
        self.raw_genome = flat_weights

    def _continuous_to_material(self, val):
        """Map continuous value to discrete material type."""
        if val < -0.5:
            return self.EMPTY
        elif val < 0.5:
            return self.SOFT
        elif val < 1.5:
            return self.RIGID
        elif val < 2.5:
            return self.H_ACT
        else:
            return self.V_ACT

    def _largest_connected_component(self, body):
        """Keep only the largest connected component of non-empty voxels."""
        rows, cols = body.shape
        visited = np.zeros_like(body, dtype=bool)
        components = []

        for r in range(rows):
            for c in range(cols):
                if body[r, c] != self.EMPTY and not visited[r, c]:
                    # BFS
                    component = []
                    queue = [(r, c)]
                    visited[r, c] = True
                    while queue:
                        cr, cc = queue.pop(0)
                        component.append((cr, cc))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if (0 <= nr < rows and 0 <= nc < cols
                                    and not visited[nr, nc]
                                    and body[nr, nc] != self.EMPTY):
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                    components.append(component)

        if not components:
            return body

        largest = max(components, key=len)
        largest_set = set(largest)
        result = np.full_like(body, self.EMPTY)
        for r, c in largest_set:
            result[r, c] = body[r, c]
        return result

    def decode_body(self):
        """
        Convert continuous genome to discrete material grid.

        Returns:
            body: 2D numpy array of material types
            connections: 2D numpy array of connection info (same shape)
            is_valid: bool (has at least one actuator and one non-empty voxel)
        """
        raw = np.array([self._continuous_to_material(v) for v in self.raw_genome])
        body = raw.reshape(self.grid_shape)

        # Keep only largest connected component
        body = self._largest_connected_component(body)

        # Connections: all non-empty voxels are connected to neighbors
        connections = np.zeros_like(body)
        rows, cols = body.shape
        for r in range(rows):
            for c in range(cols):
                if body[r, c] != self.EMPTY:
                    connections[r, c] = 1

        has_actuator = np.any((body == self.H_ACT) | (body == self.V_ACT))
        has_voxel = np.any(body != self.EMPTY)
        is_valid = bool(has_actuator and has_voxel)

        return body, connections, is_valid

    def get_actuator_phases(self, body):
        """
        Compute spatial phase offsets for each actuator voxel.

        Returns:
            list of phase values (one per actuator, in row-major order)
        """
        rows, cols = body.shape
        phases = []
        for r in range(rows):
            for c in range(cols):
                if body[r, c] in (self.H_ACT, self.V_ACT):
                    phase = (r / max(rows - 1, 1)) * np.pi
                    phases.append(phase)
        return phases
