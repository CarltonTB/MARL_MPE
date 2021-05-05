# Naive Implementation of prioritized experience replay
# Could be implemented more efficiently with a SumTree data structure

from transition_memory import TransitionMemory
from collections import deque
import torch
import numpy as np


class PrioritizedReplayMemory:
    def __init__(self, max_length=100000, alpha=0.6, beta=0.4, beta_annealing_steps=1000000):
        self.memory = deque(maxlen=max_length)
        self.max_length = max_length
        # hyper param used to determine how much TD error determines prioritization of a transition experience
        self.alpha = alpha
        # hyper param for importance sampling that is decayed over time throughout training
        self.beta = beta
        self.min_beta = beta
        self.max_beta = 1.0
        # number of times to anneal beta toward 1.0
        self.beta_annealing_steps = beta_annealing_steps
        # stores the temporal difference errors for the corresponding transition experience at each index
        self.priorities = np.array([], dtype=np.float32)
        # self.priorities = np.zeros(max_length, dtype=np.float32)

    def push(self, transition_memory):
        """
        :param transition_memory: TransitionMemory object from a single agent interaction with the environment
        :return: None
        """
        assert (isinstance(transition_memory, TransitionMemory))
        self.memory.append(transition_memory)
        # Priority starts out as max priority and is updated in the training function
        if len(self.memory) > 0:
            max_priority = self.priorities.max() if self.priorities.size > 0 else 1.0
            self.priorities = np.append(self.priorities, [max_priority])
        else:
            self.priorities = np.append(self.priorities, [1.0])

    def update_priorities(self, selected_indices, new_priorities):
        """
        :param selected_indices: Indices of experiences and priorities that were selected for a batch update
        :param new_priorities: New values of the priorities for those selected indices
        :return:
        """
        j = 0
        for i in selected_indices:
            self.priorities[i] = new_priorities[j]
            j += 1

    def anneal_beta(self, frame):
        if frame > self.beta_annealing_steps:
            self.beta = self.max_beta
        else:
            self.beta = self.min_beta + (frame / self.beta_annealing_steps) * (self.max_beta - self.min_beta)

    def sample(self, n):
        """
        :param n: number of
        :return: tuple of tensors containing states, actions, rewards, next states, and dones
        """
        if len(self.memory) < self.priorities.size:
            # Truncate the priorities array so it matches the size of the memory array at max capacity
            self.priorities = self.priorities[self.priorities.size-self.max_length:]
            assert(len(self.memory) == self.priorities.size)

        selection_probabilities = self.priorities**self.alpha
        selection_probabilities = selection_probabilities / selection_probabilities.sum()

        selected_indices = np.random.choice(len(self.memory), size=n, replace=False, p=selection_probabilities)
        sampled = [self.memory[i] for i in selected_indices]
        total_experiences = len(self.memory)
        importance_sampling_weights = ((1/total_experiences) * (1/selection_probabilities[selected_indices]))**self.beta
        # Normalize the weights to range (0, 1) inclusive
        importance_sampling_weights = importance_sampling_weights / np.max(importance_sampling_weights)

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for transition_memory in sampled:
            states.append(transition_memory.state)
            actions.append(transition_memory.action)
            rewards.append([transition_memory.reward])
            next_states.append(transition_memory.next_state)
            dones.append([transition_memory.done])

        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(next_states, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.int8),
                torch.tensor(importance_sampling_weights, dtype=torch.float32),
                selected_indices)


if __name__ == "__main__":
    test = PrioritizedReplayMemory()
    for i in range(test.beta_annealing_steps):
        print(test.beta)
        test.anneal_beta(i)
