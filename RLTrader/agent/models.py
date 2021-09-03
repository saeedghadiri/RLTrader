import pandas as pd
import logging
import os
import numpy as np
import tensorflow as tf
import random
import gym
from tqdm import trange
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from RLTrader.agent.buffer import ReplayBuffer
from RLTrader.agent.utils import OUActionNoise, Tensorboard
from RLTrader.apps.rltrader.config import KERNEL_INITIALIZER, GAMMA, RHO, STD_NOISE, BUFFER_SIZE, BATCH_SIZE, CRITIC_LR, \
    ACTOR_LR, TF_LOG_DIR, CHECKPOINTS_PATH, TOTAL_EPISODES, UNBALANCE_P, RENDER_ENV, SAVE_WEIGHTS, LOAD_LAST, EPS_GREEDY

sns.set_style('darkgrid')


def LSTMActorNetwork(state_space, num_actions):
    """
    Get Actor Network with the given parameters.
    Args:
        num_states: number of states in the NN
        num_actions: number of actions in the NN
    Returns:
        the Keras Model
    """

    inputs_market = tf.keras.layers.Input(shape=state_space[0], dtype=tf.float32)

    out_market = tf.keras.layers.LSTM(units=10)(inputs_market)
    out_market = tf.keras.layers.BatchNormalization()(out_market)
    out_market = tf.keras.activations.relu(out_market)

    inputs_portfo = tf.keras.layers.Input(shape=state_space[1], dtype=tf.float32)
    out_portfo = tf.keras.layers.Dense(20, kernel_initializer=KERNEL_INITIALIZER)(inputs_portfo)
    out_portfo = tf.keras.layers.BatchNormalization()(out_portfo)
    out_portfo = tf.keras.activations.relu(out_portfo)

    out = tf.keras.layers.Concatenate()([out_market, out_portfo])
    out = tf.keras.layers.Dense(50, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(out)
    outputs = tf.keras.layers.Dense(num_actions, activation="softmax")(out)

    model = tf.keras.Model([inputs_market, inputs_portfo], outputs)
    model.summary()
    return model


def LSTMCriticNetwork(state_space, num_actions):
    """
    Get Critic Network with the given parameters.
    Args:
        num_states: number of states in the NN
        num_actions: number of actions in the NN
    Returns:
        the Keras Model
    """
    last_init = tf.random_normal_initializer(stddev=0.00005)

    # State as input
    inputs_market = tf.keras.layers.Input(shape=state_space[0], dtype=tf.float32)

    out_market = tf.keras.layers.LSTM(units=10)(inputs_market)
    out_market = tf.keras.layers.BatchNormalization()(out_market)
    out_market = tf.keras.activations.relu(out_market)

    inputs_portfo = tf.keras.layers.Input(shape=state_space[1], dtype=tf.float32)
    out_portfo = tf.keras.layers.Dense(20, kernel_initializer=KERNEL_INITIALIZER)(inputs_portfo)
    out_portfo = tf.keras.layers.BatchNormalization()(out_portfo)
    out_portfo = tf.keras.activations.relu(out_portfo)

    state_out = tf.keras.layers.Concatenate()([out_market, out_portfo])

    # Action as input
    action_input = tf.keras.layers.Input(shape=(num_actions), dtype=tf.float32)
    action_out = tf.keras.layers.Dense(30, activation=tf.nn.leaky_relu,
                                       kernel_initializer=KERNEL_INITIALIZER)(
        action_input / 1)

    # Both are passed through separate layer before concatenating
    added = tf.keras.layers.Add()([state_out, action_out])

    added = tf.keras.layers.BatchNormalization()(added)
    outs = tf.keras.layers.Dense(70, activation=tf.nn.leaky_relu,
                                 kernel_initializer=KERNEL_INITIALIZER)(added)
    outs = tf.keras.layers.BatchNormalization()(outs)
    outputs = tf.keras.layers.Dense(1, kernel_initializer=last_init)(outs)

    # Outputs single value for give state-action
    model = tf.keras.Model([inputs_market, inputs_portfo, action_input], outputs)
    model.summary()
    return model


def CNNActorNetwork(state_space, num_actions):
    n_sequence = state_space[0][0]
    inputs_market = tf.keras.layers.Input(shape=state_space[0], dtype=tf.float32)

    out_market = tf.keras.layers.Conv2D(filters=2, activation='relu', kernel_size=(1, 3), strides=(1, 1),
                                        padding='same')(inputs_market)
    out_market = tf.keras.layers.Conv2D(filters=20, activation='relu', kernel_size=(1, n_sequence - 2),
                                        strides=(n_sequence, 1), padding='same')(out_market)

    inputs_portfo = tf.keras.layers.Input(shape=state_space[1], dtype=tf.float32)
    out_portfo = tf.keras.layers.Reshape((1, state_space[1][0], 1))(inputs_portfo)

    out = tf.keras.layers.Concatenate(axis=3)([out_portfo, out_market])

    out = tf.keras.layers.Conv2D(filters=1, activation='relu', kernel_size=(1, 1), strides=(1, 1),
                                 padding='same')(out)
    out = tf.keras.layers.Flatten()(out)

    cash_bias = tf.Variable([[0.]], shape=[1, 1], name='cash_bias', dtype=tf.float32, trainable=False)
    cash_bias = tf.keras.layers.Dense(1)(cash_bias)
    cash_bias = tf.tile(cash_bias, [tf.shape(inputs_market)[0], 1])

    out = tf.keras.layers.Concatenate()([out, cash_bias])
    outputs = tf.keras.activations.softmax(out)

    model = tf.keras.Model([inputs_market, inputs_portfo], outputs)
    model.summary()
    return model


def CNNCriticNetwork(state_space, num_actions):
    n_sequence = state_space[0][0]
    inputs_market = tf.keras.layers.Input(shape=state_space[0], dtype=tf.float32)

    out_market = tf.keras.layers.Conv2D(filters=2, activation='relu', kernel_size=(1, 3), strides=(1, 1),
                                        padding='same')(inputs_market)
    out_market = tf.keras.layers.Conv2D(filters=20, activation='relu', kernel_size=(1, n_sequence - 2),
                                        strides=(n_sequence, 1), padding='same')(out_market)

    inputs_portfo = tf.keras.layers.Input(shape=state_space[1], dtype=tf.float32)
    out_portfo = tf.keras.layers.Reshape((1, state_space[1][0], 1))(inputs_portfo)

    out = tf.keras.layers.Concatenate(axis=3)([out_portfo, out_market])

    out = tf.keras.layers.Conv2D(filters=1, activation='relu', kernel_size=(1, 1), strides=(1, 1),
                                 padding='same')(out)
    out = tf.keras.layers.Flatten()(out)

    action_input = tf.keras.layers.Input(shape=(num_actions), dtype=tf.float32)

    outs = tf.keras.layers.Concatenate()([out, action_input])

    outputs = tf.keras.layers.Dense(1)(outs)

    # Outputs single value for give state-action
    model = tf.keras.Model([inputs_market, inputs_portfo, action_input], outputs)

    model.summary()
    return model


def update_target(model_target, model_ref, rho=0):
    """
    Update target's weights with the given model reference
    Args:
        model_target: the target model to be changed
        model_ref: the reference model
        rho: the ratio of the new and old weights
    """
    model_target.set_weights([rho * ref_weight + (1 - rho) * target_weight
                              for (target_weight, ref_weight) in
                              list(zip(model_target.get_weights(), model_ref.get_weights()))])


class Brain:
    """
    The Brain that contains all the models
    """

    def __init__(self, state_space, action_space, dnn_type, gamma=GAMMA, rho=RHO,
                 std_noise=STD_NOISE):

        state_space = [state_space['market'].shape, state_space['portfo'].shape]
        num_actions = action_space.shape[0]
        # initialize everything
        if dnn_type == 'LSTM':
            self.actor_network = LSTMActorNetwork(state_space, num_actions)
            self.critic_network = LSTMCriticNetwork(state_space, num_actions)
            self.actor_target = LSTMActorNetwork(state_space, num_actions)
            self.critic_target = LSTMCriticNetwork(state_space, num_actions)
        elif dnn_type == 'CNN':
            self.actor_network = CNNActorNetwork(state_space, num_actions)
            self.critic_network = CNNCriticNetwork(state_space, num_actions)
            self.actor_target = CNNActorNetwork(state_space, num_actions)
            self.critic_target = CNNCriticNetwork(state_space, num_actions)

        # Making the weights equal initially
        self.actor_target.set_weights(self.actor_network.get_weights())
        self.critic_target.set_weights(self.critic_network.get_weights())

        self.buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.gamma = tf.constant(gamma)
        self.rho = rho

        self.num_actions = num_actions
        self.std_noise = std_noise

        # optimizers
        self.critic_optimizer = tf.keras.optimizers.Adam(CRITIC_LR, amsgrad=True)
        self.actor_optimizer = tf.keras.optimizers.Adam(ACTOR_LR, amsgrad=True)

        # temporary variable for side effects
        self.cur_action = None

        # define update weights with tf.function for improved performance
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(None, *state_space[0]), dtype=tf.float32),
                tf.TensorSpec(shape=(None, *state_space[1]), dtype=tf.float32),
                tf.TensorSpec(shape=(None, num_actions), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, *state_space[0]), dtype=tf.float32),
                tf.TensorSpec(shape=(None, *state_space[1]), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            ])
        def update_weights(s_0, s_1, a, r, sn_0, sn_1, d):
            """
            Function to update weights with optimizer
            """
            with tf.GradientTape() as tape:
                # define target
                # y = r + self.gamma * (1 - d) * self.critic_target([sn_0, sn_1, self.actor_target([sn_0, sn_1])])
                # because we dont have a terminal state
                y = r + self.gamma * self.critic_target([sn_0, sn_1, self.actor_target([sn_0, sn_1])])
                # define the delta Q
                critic_loss = tf.math.reduce_mean(tf.math.abs(y - self.critic_network([s_0, s_1, a])))
            critic_grad = tape.gradient(critic_loss, self.critic_network.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic_network.trainable_variables))

            with tf.GradientTape() as tape:
                # define the delta mu
                actor_loss = -tf.math.reduce_mean(self.critic_network([s_0, s_1, self.actor_network([s_0, s_1])]))
            actor_grad = tape.gradient(actor_loss, self.actor_network.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor_network.trainable_variables))
            return critic_loss, actor_loss

        self.update_weights = update_weights

    def act(self, state, _notrandom=True, noise=True):
        """
        Run action by the actor network
        Args:
            state: the current state
            _notrandom: whether greedy is used
            noise: whether noise is to be added to the result action (this improves exploration)
        Returns:
            the resulting action
        """
        state_ = [state[0][np.newaxis, ...], state[1][np.newaxis, ...]]
        if _notrandom:
            self.cur_action = self.actor_network(state_)[0].numpy()
        else:
            self.cur_action = self.actor_network(state_)[0].numpy()
            noise_action = np.random.randn(len(self.cur_action)) * self.std_noise if noise else 0
            self.cur_action = np.abs(self.cur_action + noise_action)
            self.cur_action = self.cur_action / np.sum(self.cur_action)

        return self.cur_action

    def remember(self, prev_state, reward, state, done):
        """
        Store states, reward, done value to the buffer
        """
        # record it in the buffer based on its reward
        self.buffer.append(prev_state, self.cur_action, reward, state, done)

    def learn(self, entry):
        """
        Run update for all networks (for training)
        """
        s_0, s_1, a, r, sn_0, sn_1, d = zip(*entry)
        if self.buffer.get_buffer_size() < 100:
            return 0, 0
        c_l, a_l = self.update_weights(tf.convert_to_tensor(s_0, dtype=tf.float32),
                                       tf.convert_to_tensor(s_1, dtype=tf.float32),
                                       tf.convert_to_tensor(a, dtype=tf.float32),
                                       tf.convert_to_tensor(r, dtype=tf.float32),
                                       tf.convert_to_tensor(sn_0, dtype=tf.float32),
                                       tf.convert_to_tensor(sn_1, dtype=tf.float32),
                                       tf.convert_to_tensor(d, dtype=tf.float32))

        update_target(self.actor_target, self.actor_network, self.rho)
        update_target(self.critic_target, self.critic_network, self.rho)

        return c_l, a_l

    def save_weights(self, path):
        """
        Save weights to `path`
        """
        parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        # Save the weights
        self.actor_network.save_weights(path + "an.h5")
        self.critic_network.save_weights(path + "cn.h5")
        self.critic_target.save_weights(path + "ct.h5")
        self.actor_target.save_weights(path + "at.h5")

    def load_weights(self, path):
        """
        Load weights from path
        """
        try:
            self.actor_network.load_weights(path + "an.h5")
            self.critic_network.load_weights(path + "cn.h5")
            self.critic_target.load_weights(path + "ct.h5")
            self.actor_target.load_weights(path + "at.h5")
        except OSError as err:
            logging.warning("Weights files cannot be found, %s", err)


class Agent:
    def __init__(self, env, env_test, dnn_type):
        self.brain = Brain(env.observation_space, env.action_space, dnn_type)
        self.env = env
        self.env_test = env_test
        self.dnn_type = dnn_type

    def learn(self):

        tensorboard = Tensorboard(log_dir=TF_LOG_DIR)

        # load weights if available
        logging.info("Loading weights from %s*, make sure the folder exists", CHECKPOINTS_PATH)
        if LOAD_LAST:
            self.brain.load_weights(CHECKPOINTS_PATH)

        # all the metrics

        q_loss = []
        a_loss = []

        # run iteration

        for ep in range(TOTAL_EPISODES):
            prev_state = self.env.reset(random_start=True)
            done = False

            while not done:
                if RENDER_ENV:  # render the environment into GUI
                    self.env.render()

                # Recieve state and reward from environment.
                cur_act = self.brain.act(prev_state, _notrandom=random.random() < ep / TOTAL_EPISODES, noise=True)

                state, reward, done, _ = self.env.step(cur_act)
                self.brain.remember(prev_state, reward, state, int(done))

                # update weights
                c, a = self.brain.learn(self.brain.buffer.get_batch(unbalance_p=UNBALANCE_P))

                q_loss.append(c)
                a_loss.append(a)

                prev_state = state

            # perform one evaluation on train and test environments

            train_reward, _, train_asset = self.run(self.env, 'train_actions')
            test_reward, _, test_asset = self.run(self.env_test, 'test_actions')

            # print the average reward
            tensorboard(ep, np.mean(train_reward), np.mean(test_reward), np.mean(q_loss), np.mean(a_loss),
                        (test_asset / self.env.initial_asset) - 1, self.brain.buffer.get_buffer_size())

            # save weights
            if ep % 5 == 0 and SAVE_WEIGHTS:
                self.brain.save_weights(CHECKPOINTS_PATH)

        self.brain.save_weights(CHECKPOINTS_PATH)

        logging.info("Training done...")

    def run(self, env, action_plot_name=None):
        state = env.reset()
        done = False
        all_action = []
        all_reward = []
        while not done:
            cur_act = self.brain.act(state, _notrandom=True, noise=False)
            state, reward, done, _ = env.step(cur_act)
            all_reward.append(reward)
            all_action.append(cur_act)

        if action_plot_name is not None:
            all_action = pd.DataFrame(all_action)
            all_action.plot(figsize=(20, 7))
            plt.savefig(os.path.join('{}.png'.format(action_plot_name)))

        return all_reward, all_action, env.asset
