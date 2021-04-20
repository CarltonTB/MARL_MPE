import random


class RandomAgent:

    def policy(self, observation, done):
        """Return action based on the current state and policy"""
        if done:
            return None
        else:
            return random.randint(0, 4)

