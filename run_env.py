from pettingzoo.mpe import simple_adversary_v2
import random


def run_sa():
    env = simple_adversary_v2.env(N=2, max_cycles=25)
    for episode in range(100):
        env.reset()
        for agent in env.agent_iter():
            observation, reward, done, info = env.last()
            if done:
                action = None
            else:
                action = random.randint(0, 4)
            env.step(action)
            env.render()
    env.close()


if __name__ == "__main__":
    run_sa()


