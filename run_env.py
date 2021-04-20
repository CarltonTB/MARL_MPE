from pettingzoo.mpe import simple_adversary_v2
from random_agent import RandomAgent

agent_dict = {
    "agent_0": RandomAgent(),
    "agent_1": RandomAgent(),
    "adversary_0": RandomAgent(),
}


def run_simple_adversary():
    env = simple_adversary_v2.env(N=2, max_cycles=25)
    for episode in range(100):
        env.reset()
        for agent in env.agent_iter():
            observation, reward, done, info = env.last()
            action = agent_dict.get(agent).policy(observation, done)
            env.step(action)
            env.render()
    env.close()


if __name__ == "__main__":
    run_simple_adversary()


