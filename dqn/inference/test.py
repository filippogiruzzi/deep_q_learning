import os
import argparse

import gym
import tensorflow as tf

from dqn.models.pre_processing import StatePreprocessor
from dqn.models.estimators import QEstimator
from dqn.models.dqn import TestDQNAgent


def main():
    parser = argparse.ArgumentParser(description='Test DQN to play Space Invaders')
    parser.add_argument('--exp-dir', '-exp', type=str, default='', help='dir. to record experiments')
    parser.add_argument('--n-eps', '-n', type=int, default=10, help='number of episodes')
    args = parser.parse_args()

    env = gym.envs.make("SpaceInvaders-v0")

    tf.reset_default_graph()
    base_dir = './'
    if args.exp_dir:
        base_dir = args.exp_dir
    experiment_dir = os.path.abspath("{}experiments/{}".format(base_dir, env.spec.id))

    state_preprocessor = StatePreprocessor()
    q_value_estimator = QEstimator(scope="q_estimator", summaries_dir=experiment_dir)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for t, stats in TestDQNAgent(sess,
                                     env,
                                     q_value_estimator,
                                     state_preprocessor,
                                     num_episodes=args.n_eps,
                                     experiment_dir=experiment_dir):
            print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))


if __name__ == '__main__':
    main()
