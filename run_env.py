from pettingzoo.mpe import simple_adversary_v2
from random_agent import RandomAgent
import time
import matplotlib.pyplot as plt
from dqn_agent import *


def train_simple_adversary(agent_dict, episodes, load_from_checkpoint=False):
    env = simple_adversary_v2.env(N=2, max_cycles=50)
    start = time.time()
    for episode in range(episodes):
        env.reset()
        score_dict = {
            "agent_0": 0,
            "agent_1": 0,
            "adversary_0": 0,
        }
        for agent in env.agent_iter():
            agent_obj = agent_dict.get(agent)
            observation, reward, done, info = env.last()
            action = agent_obj.policy(observation, done)
            env.step(action)

            if agent_obj.last_observation is not None and agent_obj.last_action is not None:
                agent_obj.push_memory(TransitionMemory(agent_obj.last_observation, agent_obj.last_action, reward, observation, done))

            agent_obj.last_observation = observation
            agent_obj.last_action = action
            agent_obj.do_training_update()
            agent_obj.update_target_network()
            agent_obj.decay_epsilon()
            score_dict[agent] += reward

            # agent.prioritized_memory.push(TransitionMemory(observation, action, reward, next_observation, done))
            # agent.do_prioritized_training_update((episode * 1000) + t)
            # env.render()
        if episodes % 100 == 0:
            print("Episode scores:")
            print(score_dict)

    env.close()
    print(f'Time elapsed: {time.time() - start} seconds')


def test_simple_adversary(agent_dict, episodes):
    env = simple_adversary_v2.env(N=2, max_cycles=50)
    for episodes in range(episodes):
        env.reset()
        for agent in env.agent_iter():
            observation, reward, done, info = env.last()
            action = agent_dict.get(agent).policy(observation, done)
            env.step(action)
            env.render()
    env.close()


def plot_score(score, rolling_mean, filename):
    plt.figure(figsize=(10, 5))
    plt.title("Score vs. Episodes")
    plt.plot(score, label="Score")
    plt.plot(rolling_mean, label="Mean last 100 episodes")
    plt.ylabel("Score")
    plt.xlabel("Episode")
    plt.savefig(f"./graphs/{filename}")
    plt.show()


if __name__ == "__main__":
    random_agent_dict = {
        "agent_0": RandomAgent(),
        "agent_1": RandomAgent(),
        "adversary_0": RandomAgent(),
    }
    agent_q_net = AgentDQN()
    agent_target_net = AgentDQN()
    agent_0 = DQNAgent(epsilon=1.0, min_epsilon=0.1, decay_rate=0.999,
                       learning_rate=0.0001, gamma=0.99, batch_size=64,
                       tau=0.001, q_network=agent_q_net, target_network=agent_target_net,
                       max_memory_length=100000)

    agent_1 = DQNAgent(epsilon=1.0, min_epsilon=0.1, decay_rate=0.999,
                       learning_rate=0.0001, gamma=0.99, batch_size=64,
                       tau=0.001, q_network=agent_q_net, target_network=agent_target_net,
                       max_memory_length=100000)

    adversary_0 = DQNAgent(epsilon=1.0, min_epsilon=0.1, decay_rate=0.999,
                           learning_rate=0.0001, gamma=0.99, batch_size=64,
                           tau=0.001, q_network=AdversaryDQN(), target_network=AdversaryDQN(),
                           max_memory_length=100000)
    dqn_agent_dict = {
        "agent_0": agent_0,
        "agent_1": agent_1,
        "adversary_0": adversary_0,
    }
    train_simple_adversary(dqn_agent_dict, 100, load_from_checkpoint=False)


