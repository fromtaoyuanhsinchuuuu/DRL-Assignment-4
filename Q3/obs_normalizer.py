# obs_normalizer.py
import numpy as np
import torch

class ObservationNormalizer:
    """
    Normalizes observations using running mean and standard deviation.
    This is particularly useful for high-dimensional state spaces like Humanoid.
    
    Has a training mode (updates running stats) and evaluation mode (uses fixed stats).
    """
    def __init__(self, state_dim, clip_range=10.0, device=torch.device("cpu")):
        """
        Initialize the observation normalizer.
        
        Args:
            state_dim (int): Dimension of state space
            clip_range (float): Range to clip normalized observations to
            device (torch.device): Device to store tensors on
        """
        self.state_dim = state_dim
        self.clip_range = clip_range
        self.device = device
        self.training = True  # Flag to control whether to update statistics
        
        # Initialize running statistics
        self.running_mean = np.zeros(state_dim, dtype=np.float32)
        self.running_var = np.ones(state_dim, dtype=np.float32)
        self.count = 1e-4  # Small initial count to avoid division by zero
        
    def set_training_mode(self, training=True):
        """
        Set the normalizer to training or evaluation mode.
        In training mode, statistics are updated with new observations.
        In evaluation mode, fixed statistics are used without updates.
        
        Args:
            training (bool): Whether to update statistics
        """
        self.training = training
    
    def update(self, observations):
        """
        Update running statistics with new observations.
        Only updates if in training mode.
        
        Args:
            observations (np.ndarray): Batch of observations
        """
        # Skip update if in evaluation mode
        if not self.training:
            return
            
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)
            
        batch_size = observations.shape[0]
        batch_mean = np.mean(observations, axis=0)
        batch_var = np.var(observations, axis=0)
        
        # Update running statistics using Welford's online algorithm
        new_count = self.count + batch_size
        delta = batch_mean - self.running_mean
        self.running_mean = self.running_mean + delta * batch_size / new_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_size
        M2 = m_a + m_b + np.square(delta) * self.count * batch_size / new_count
        self.running_var = M2 / new_count
        self.count = new_count
        
    def normalize(self, observation):
        """
        Normalize an observation using running statistics.
        
        Args:
            observation (np.ndarray): Observation to normalize
            
        Returns:
            np.ndarray: Normalized observation
        """
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
            
        normalized_obs = (observation - self.running_mean) / (np.sqrt(self.running_var) + 1e-8)
        normalized_obs = np.clip(normalized_obs, -self.clip_range, self.clip_range)
        
        if normalized_obs.shape[0] == 1:
            normalized_obs = normalized_obs.flatten()
            
        return normalized_obs
    
    def normalize_torch(self, observation):
        """
        Normalize a torch observation using running statistics.
        
        Args:
            observation (torch.Tensor): Observation to normalize
            
        Returns:
            torch.Tensor: Normalized observation
        """
        # Convert numpy arrays to torch tensors
        running_mean = torch.FloatTensor(self.running_mean).to(self.device)
        running_var = torch.FloatTensor(self.running_var).to(self.device)
        
        normalized_obs = (observation - running_mean) / (torch.sqrt(running_var) + 1e-8)
        normalized_obs = torch.clamp(normalized_obs, -self.clip_range, self.clip_range)
        
        return normalized_obs
    
    def save(self, path):
        """
        Save normalizer statistics to a file.
        
        Args:
            path (str): Path to save to
        """
        np.savez(
            path,
            running_mean=self.running_mean,
            running_var=self.running_var,
            count=self.count
        )
        
    def load(self, path):
        """
        Load normalizer statistics from a file.
        
        Args:
            path (str): Path to load from
        """
        data = np.load(path)
        self.running_mean = data['running_mean']
        self.running_var = data['running_var']
        self.count = data['count']
