import os
import argparse

import gym
import tensorflow as tf

from dqn.models.pre_processing import StatePreprocessor
from dqn.models.estimators import QEstimator
from dqn.models.dqn import TrainDQNAgent


def main():
    parser = argparse.ArgumentParser(description='Train DQN to play Space Invaders')
    parser.add_argument('--exp-dir', '-exp', type=str, default='', help='dir. to record experiments')
    parser.add_argument('--n-eps', '-n', type=int, default=10000, help='number of episodes')
    parser.add_argument('--replay-size', '-rs', type=int, default=500000, help='replay memory size')
    parser.add_argument('--replay-init-size', '-ris', type=int, default=50000, help='reaply memory initial size')
    parser.add_argument('--target-update', '-t', type=int, default=10000, help='target net update delay')
    parser.add_argument('--epsilon-range', '-e', type=str, default='1.0-0.1', help='epsilon initial & final values')
    parser.add_argument('--epsilon-decay', '-d', type=int, default=500000, help='epsilon decay steps')
    parser.add_argument('--gamma', '-g', type=float, default=0.99, help='discount factor')
    parser.add_argument('--batch-size', '-bs', type=int, default=32, help='batch size')
    args = parser.parse_args()

    assert len(args.epsilon_range.split('-')) == 2, 'Wrong argument values'
    epsilon_start, epsilon_end = [float(x) for x in args.epsilon_range.split('-')]

    env = gym.envs.make("SpaceInvaders-v0")

    tf.reset_default_graph()
    base_dir = './'
    if args.exp_dir:
        base_dir = args.exp_dir
    experiment_dir = os.path.abspath("{}experiments/{}".format(base_dir, env.spec.id))

    state_preprocessor = StatePreprocessor()
    q_value_estimator = QEstimator(scope="q_estimator", summaries_dir=experiment_dir)
    target_net_estimator = QEstimator(scope="target_q")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for t, stats in TrainDQNAgent(sess,
                                      env,
                                      q_value_estimator=q_value_estimator,
                                      target_net_estimator=target_net_estimator,
                                      state_preprocessor=state_preprocessor,
                                      experiment_dir=experiment_dir,
                                      num_episodes=args.n_eps,
                                      replay_memory_size=args.replay_size,
                                      replay_memory_init_size=args.replay_init_size,
                                      target_net_update=args.target_update,
                                      epsilon_start=epsilon_start,
                                      epsilon_end=epsilon_end,
                                      epsilon_decay_steps=args.epsilon_decay,
                                      gamma=args.gamma,
                                      batch_size=args.batch_size,
                                      record_steps=50):
            print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))


if __name__ == '__main__':
    main()
