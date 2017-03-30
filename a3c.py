from __future__ import print_function
from collections import namedtuple
import numpy as np
import tensorflow as tf
from model import LSTMPolicy
import six.moves.queue as queue
import scipy.signal
import threading
import distutils.version
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')
import cv2


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class A3C(object):
    def __init__(self, env, task, visualise, test=False):
        """
An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
should be computed.
"""

        self.env = env
        self.task = task
        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        if test:
           worker_device = "/job:eval/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = LSTMPolicy(env.observation_space.shape, env.action_space.n)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = LSTMPolicy(env.observation_space.shape, env.action_space.n)
                pi.global_step = self.global_step

            self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")

            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf = tf.nn.softmax(pi.logits)

            # the "policy gradients" loss:  its derivative is precisely the policy gradient
            # notice that self.ac is a placeholder that is provided externally.
            # adv will contain the advantages, as calculated in process_rollout
            pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

            # loss of value function
            vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))
            entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

            bs = tf.to_float(tf.shape(pi.x)[0])
            self.loss = pi_loss + 0.5 * vf_loss - entropy * 0.01


            self.visualise = visualise

            grads = tf.gradients(self.loss, pi.var_list)

            if use_tf12_api:
                tf.summary.scalar("model/policy_loss", pi_loss / bs)
                tf.summary.scalar("model/value_loss", vf_loss / bs)
                tf.summary.scalar("model/entropy", entropy / bs)
                tf.summary.image("model/state", pi.x)
                tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
                tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
                self.summary_op = tf.summary.merge_all()

            else:
                tf.scalar_summary("model/policy_loss", pi_loss / bs)
                tf.scalar_summary("model/value_loss", vf_loss / bs)
                tf.scalar_summary("model/entropy", entropy / bs)
                tf.image_summary("model/state", pi.x)
                tf.scalar_summary("model/grad_global_norm", tf.global_norm(grads))
                tf.scalar_summary("model/var_global_norm", tf.global_norm(pi.var_list))
                self.summary_op = tf.merge_all_summaries()

            grads, _ = tf.clip_by_global_norm(grads, 40.0)

            # This is sync ops which copy weights from shared space to the local.
            self.sync = tf.group(
                *(
                    [ v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)]
                 ))


            grads_and_vars = list(zip(grads, self.network.var_list))
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])
            # each worker has a different set of adam optimizer parameters
            opt = tf.train.AdamOptimizer(1e-4)
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
            self.summary_writer = None
            self.local_steps = 0




    def start(self, sess, summary_writer):
        self.summary_writer = summary_writer

        # Initialise last_state and last_features
        self.last_state = self.env.reset()
        self.last_features = self.local_network.get_initial_features()
        self.last_action = np.zeros(self.env.action_space.n)
        self.last_reward = [0]
        self.length = 0
        self.rewards = 0

    def process(self, sess):
        """
        Every time process is called.
        The network get sync.
        The environment is run for 20 steps or until termination.
        The worker calculates gradients and then one update to the shared weight is made.
        (one local step = one update  =< 20 env steps )
        (global step is the number of frames)
        """
        sess.run(self.sync)  # copy weights from shared to local

        # Environment run for 20 steps or less
        terminal_end = False
        num_local_steps = 20
        env = self.env
        policy = self.local_network

        states  = []
        actions = []
        rewards = []
        values  = []
        r       = 0.0
        terminal= False
        features= []
        prev_actions = []
        prev_rewards = []

        # TODO: add intrinsic reward see if it works
        goal_patch = np.zeros([84,84,3])
        goal_patch[45:55, 55:65, :] = 1

        for _local_step in range(num_local_steps):
            # Take a step
            fetched = policy.act(self.last_state, self.last_features[0], self.last_features[1], self.last_action, self.last_reward)
            action, value_, features_ = fetched[0], fetched[1], fetched[2:]
            # argmax to convert from one-hot
            state, reward, terminal, info = env.step(action.argmax())


            # Intrinsic reward
            pixel_changes = (state - self.last_state)**2
            # mean square error normalized by all pixel_changes
            intrinsic_reward = np.sum( pixel_changes * goal_patch ) / np.sum( pixel_changes + 1e-5)
            # Apply intrinsic reward
            reward += intrinsic_reward

            if self.visualise:
                vis = 0.5 * state + 0.5 * goal_patch
                vis = cv2.resize(vis, (500,500))
                cv2.imshow('img', vis)
                cv2.waitKey(10)

            # collect the experience
            states += [self.last_state]
            actions += [action]
            rewards += [reward]
            values += [value_]
            features += [self.last_features]
            prev_actions += [self.last_action]
            prev_rewards += [self.last_reward]

            self.length += 1
            self.rewards += reward

            self.last_state = state
            self.last_features = features_
            self.last_action = action
            self.last_reward = [reward]

            if info:
                summary = tf.Summary()
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))

                if _local_step == 19:
                    summary.value.add(tag='global/shaped_reward_per_time' , simple_value=np.mean(rewards) )

                self.summary_writer.add_summary(summary, policy.global_step.eval())
                self.summary_writer.flush()

            timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            if terminal or self.length >= timestep_limit:
                terminal_end = True
                if self.length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    self.last_state = env.reset()
                self.last_features = policy.get_initial_features()
                print("Episode finished. Sum of rewards: %d. Length: %d" % (self.rewards, self.length))

                summary = tf.Summary()
                summary.value.add(tag='global/episode_shaped_reward', simple_value=self.rewards)
                self.summary_writer.add_summary(summary, policy.global_step.eval())
                self.summary_writer.flush()

                self.length = 0
                self.rewards = 0
                break

        if not terminal_end:
            r = policy.value(self.last_state, self.last_features[0], self.last_features[1], self.last_action, self.last_reward)

        # Process rollout
        gamma = 0.99
        lambda_ = 1.0
        batch_si = np.asarray(states)
        batch_a = np.asarray(actions)
        rewards_plus_v = np.asarray(rewards + [r])
        rewards = np.asarray(rewards)
        vpred_t = np.asarray(values + [r])
        batch_r = discount(rewards_plus_v, gamma)[:-1]
        delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
        # this formula for the advantage comes "Generalized Advantage Estimation":
        # https://arxiv.org/abs/1506.02438
        batch_adv = discount(delta_t, gamma * lambda_)
        batch_prev_a = np.asarray(prev_actions)
        batch_prev_r = np.asarray(prev_rewards)

        # Gradient Calculation
        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0
        if should_compute_summary:
            fetches = [self.summary_op, self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]

        features = features[0] # only use first feature into dynamic rnn

        feed_dict = {
            self.local_network.x: batch_si,
            self.ac: batch_a,
            self.adv: batch_adv,
            self.r: batch_r,
            self.local_network.state_in[0]: features[0],
            self.local_network.state_in[1]: features[1],
            self.local_network.prev_action: batch_prev_a,
            self.local_network.prev_reward: batch_prev_r
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()
        self.local_steps += 1

    def evaluate(self,sess):
        global_step = sess.run(self.global_step)
        sess.run(self.sync)
        policy = self.local_network
        env = self.env
        rewards_stat = []
        length_stat = []
        # average over 40 episode?
        for episode in range(40):
            terminal = False
            last_state = env.reset()
            last_features = policy.get_initial_features()
            rewards = 0
            length = 0
            while not terminal:
                fetched = policy.act(last_state, *last_features)
                action, value_, features_ = fetched[0], fetched[1], fetched[2:]
                state, reward, terminal, info = env.step(action.argmax())
                if self.visualise:
                    env.render()
                length += 1
                rewards += reward
                last_state = state
                last_features = features_

            rewards_stat.append(rewards)
            length_stat.append(length)

        summary = tf.Summary()
        summary.value.add(tag='Eval/Average_Reward', simple_value=np.mean(rewards_stat))
        summary.value.add(tag='Eval/SD_Reward', simple_value=np.std(rewards_stat))
        summary.value.add(tag='Eval/Average_Lenght', simple_value=np.mean(length_stat))
        self.summary_writer.add_summary(summary, global_step)
        self.summary_writer.flush()
