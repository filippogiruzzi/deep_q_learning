import os
import sys

import tensorflow as tf
import numpy as np
import itertools
import random
from gym.wrappers import Monitor
from collections import namedtuple

from dqn.models.estimators import TargetNetUpdate


VALID_ACTIONS = [0, 1, 2, 3]


def make_epsilon_greedy_policy(estimator, num_actions):
    def policy_fn(sess, observation, epsilon):
        action_probs = np.ones(num_actions, dtype=float) * epsilon / (num_actions - 1)
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        action_probs[best_action] = 1.0 - epsilon
        return action_probs
    return policy_fn


def TrainDQNAgent(sess,
                  env,
                  q_value_estimator,
                  target_net_estimator,
                  state_preprocessor,
                  num_episodes,
                  experiment_dir,
                  replay_memory_size=500000,
                  replay_memory_init_size=50000,
                  target_net_update=10000,
                  epsilon_start=1.0,
                  epsilon_end=0.1,
                  epsilon_decay_steps=500000,
                  gamma=0.99,
                  batch_size=32,
                  record_steps=50):

    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
    EpisodeStats = namedtuple('Stats', ['episode_lengths', 'episode_rewards'])
    stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))
    replay_memory = []
    estimator_copy = TargetNetUpdate(q_value_estimator, target_net_estimator)

    ckpt_dir = os.path.join(experiment_dir, 'checkpoints')
    ckpt_path = os.path.join(ckpt_dir, 'model')
    record_path = os.path.join(experiment_dir, 'record')

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(record_path):
        os.makedirs(record_path)

    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    if latest_checkpoint:
        print('\nLoading model checkpoint {}...'.format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    total_t = sess.run(tf.contrib.framework.get_global_step())
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    policy = make_epsilon_greedy_policy(q_value_estimator, len(VALID_ACTIONS))

    print('\nPopulating replay memory...')
    state = env.reset()
    state = state_preprocessor.process(sess, state)
    state = np.stack([state] * 4, axis=2)
    for i in range(replay_memory_init_size):
        action_probs = policy(sess, state, epsilons[min(total_t, epsilon_decay_steps - 1)])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
        next_state = state_preprocessor.process(sess, next_state)
        next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
        replay_memory.append(Transition(state, action, reward, next_state, done))
        print('\r{}/{}'.format(len(replay_memory), replay_memory_init_size), end="")
        sys.stdout.flush()
        if done:
            state = env.reset()
            state = state_preprocessor.process(sess, state)
            state = np.stack([state] * 4, axis=2)
        else:
            state = next_state

    env = Monitor(env, directory=record_path, video_callable=lambda count: count % record_steps == 0, resume=True)
    for i_episode in range(num_episodes):
        saver.save(tf.get_default_session(), ckpt_path)

        state = env.reset()
        state = state_preprocessor.process(sess, state)
        state = np.stack([state] * 4, axis=2)
        loss = None

        for t in itertools.count():
            # env.render()
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]

            if total_t % target_net_update == 0:
                estimator_copy.update(sess)
                print("\nCopied model parameters to target network.")

            print("\rStep {} ({}) | Episode {}/{} | loss: {}".format(t, total_t, i_episode + 1, num_episodes, loss),
                  end="")
            sys.stdout.flush()

            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
            next_state = state_preprocessor.process(sess, next_state)
            next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)

            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            replay_memory.append(Transition(state, action, reward, next_state, done))

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            q_values_next = target_net_estimator.predict(sess, next_states_batch)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * gamma * np.amax(
                q_values_next, axis=1)

            states_batch = np.array(states_batch)
            loss = q_value_estimator.update(sess, states_batch, action_batch, targets_batch)

            if done:
                break

            state = next_state
            total_t += 1

        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=epsilon, tag="episode/epsilon")
        episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], tag="episode/reward")
        episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], tag="episode/length")
        q_value_estimator.summary_writer.add_summary(episode_summary, i_episode)
        q_value_estimator.summary_writer.flush()

        episode_stats = EpisodeStats(episode_lengths=stats.episode_lengths[:i_episode + 1],
                                     episode_rewards=stats.episode_rewards[:i_episode + 1])
        yield total_t, episode_stats

    return stats


def TestDQNAgent(sess,
                 env,
                 q_value_estimator,
                 state_preprocessor,
                 num_episodes,
                 experiment_dir,
                 record_steps=1):

    EpisodeStats = namedtuple('Stats', ['episode_lengths', 'episode_rewards'])
    stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

    ckpt_dir = os.path.join(experiment_dir, 'checkpoints')
    record_path = os.path.join(experiment_dir, 'record/tests/')

    if not os.path.exists(record_path):
        os.makedirs(record_path)

    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    if latest_checkpoint:
        print('\nLoading model checkpoint {}...'.format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    total_t = sess.run(tf.contrib.framework.get_global_step())
    epsilon = 0.1
    policy = make_epsilon_greedy_policy(q_value_estimator, len(VALID_ACTIONS))

    env = Monitor(env, directory=record_path, video_callable=lambda count: count % record_steps == 0, resume=True)
    for i_episode in range(num_episodes):
        state = env.reset()
        state = state_preprocessor.process(sess, state)
        state = np.stack([state] * 4, axis=2)

        for t in itertools.count():
            env.render()

            print("\rStep {} ({}) | Episode {}/{}".format(t, total_t, i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
            next_state = state_preprocessor.process(sess, next_state)
            next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if done:
                break

            state = next_state
            total_t += 1

        episode_stats = EpisodeStats(episode_lengths=stats.episode_lengths[:i_episode + 1],
                                     episode_rewards=stats.episode_rewards[:i_episode + 1])
        yield total_t, episode_stats

    return stats
