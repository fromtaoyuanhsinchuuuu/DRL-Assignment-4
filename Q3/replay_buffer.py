# replay_buffer.py
import numpy as np
import torch

class ReplayBuffer:
    """
    Experience replay buffer for off-policy RL algorithms.
    Stores transitions (s, a, r, s', done, real_done) and allows sampling random batches.
    Distinguishes between episode termination due to failure/success (real_done=True) 
    and termination due to timeout (real_done=False).
    """
    def __init__(self, capacity, state_dim, action_dim, device):
        """
        Initialize the replay buffer.
        
        Args:
            capacity (int): Maximum number of transitions to store
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            device (torch.device): Device to store tensors on
        """
        self.capacity = capacity
        self.device = device
        
        # Preallocate memory for all arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)  # Any episode termination (timeout or real)
        self.real_dones = np.zeros((capacity, 1), dtype=np.float32)  # Only real terminations (failure/success)
        
        self.ptr = 0  # Current position in buffer
        self.size = 0  # Current size of buffer
        
    def add(self, state, action, reward, next_state, done, real_done=None):
        """
        Add a new transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode terminated (for any reason)
            real_done: Whether the episode terminated due to environment rules (not timeout)
                      If None, assumed to be the same as done
        """
        # If real_done is not provided, assume it's the same as done
        if real_done is None:
            real_done = done
            
        # Store transition in buffer
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.real_dones[self.ptr] = real_done
        
        # Update pointer and size
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones, real_dones)
        """
        # Sample random indices
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # Convert to torch tensors and move to device
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)
        real_dones = torch.FloatTensor(self.real_dones[indices]).to(self.device)
        
        return states, actions, rewards, next_states, dones, real_dones
    
    def __len__(self):
        """Return the current size of the buffer."""
        return self.size
