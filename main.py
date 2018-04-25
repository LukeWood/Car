from __future__ import division
import argparse

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


INPUT_SHAPE=(5,)
WINDOW_LENGTH=4

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env_name', default='BinaryCarRacing-v0')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)

actions = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 0]]
nb_actions=len(actions)

class CarProcessor(Processor):
    def process_observation(self, observation):
        processed_observation = np.array(observation)*10
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        return np.array(batch)

    def process_action(self, action):
        return actions[action]

    def process_reward(self, reward):
        return reward

input_shape=(WINDOW_LENGTH,) + INPUT_SHAPE

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
model = Sequential()
model.add(Flatten(input_shape=input_shape))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('relu'))
print(model.summary())

processor = CarProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.1,
                              nb_steps=10000)
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=5, gamma=.99, target_model_update=1, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]

    # Lets just keep training the same damn model
    dqn.load_weights(weights_filename)
    dqn.fit(env, callbacks=callbacks, nb_steps=10000, log_interval=5000)
    dqn.save_weights(weights_filename, overwrite=True)
    env.reset()
    dqn.test(env, nb_episodes=1, visualize=True)

    # After training is done, we save the final weights one more time.
    # Finally, evaluate our algorithm for 10 episodes.

elif args.mode == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.fit(env, nb_steps=1000000, visualize=True)
