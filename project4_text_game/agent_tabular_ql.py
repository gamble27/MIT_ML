"""Tabular QL agent"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import project4_text_game.framework as framework
import project4_text_game.utils as utils

from typing import Tuple


DEBUG = False

GAMMA = 0.5  # discounted factor
TRAINING_EP = 0.5  # epsilon-greedy parameter for training
TESTING_EP = 0.05  # epsilon-greedy parameter for testing
NUM_RUNS = 10
NUM_EPOCHS = 200
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 0.1  # learning rate for training

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)


def epsilon_greedy(state_1, state_2, q_func, epsilon) -> Tuple[int, int]:
    """Returns an action selected by an epsilon-Greedy exploration policy

    Args:
        state_1, state_2 (int, int): two indices describing the current state
        q_func (np.ndarray): current Q-function
        epsilon (float): the probability of choosing a random command

    Returns:
        (int, int): the indices describing the action/object to take
    """
    if np.random.choice([1, 0], p=[epsilon, 1 - epsilon]):
        # explore = np.random.randint(0, [NUM_ACTIONS, NUM_OBJECTS], 2)
        action_index, object_index = np.random.randint(0, NUM_ACTIONS), np.random.randint(0, NUM_OBJECTS)

        return action_index, object_index
    else:
        action_index, object_index = 0, 0
        current_q = q_func[state_1, state_2, :, :]
        max_q = current_q[0, 0]
        for a in range(NUM_ACTIONS):
            for o in range(NUM_OBJECTS):
                if current_q[a, o] > max_q:
                    max_q = current_q[a, o]
                    action_index = a
                    object_index = o

        return action_index, object_index


def tabular_q_learning(q_func, current_state_1, current_state_2, action_index,
                       object_index, reward, next_state_1, next_state_2,
                       terminal):
    """Update q_func for a given transition

    Args:
        q_func (np.ndarray): current Q-function
        current_state_1 (int): first index describing the current state
        current_state_2 (int): second index describing the current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent receives from playing current command
        next_state_1 (int): first index describing next state
        next_state_2 (int): second index describing the next state
        terminal (bool): True if this episode is over

    Returns:
        None
    """
    if not terminal:
        q_func[current_state_1, current_state_2,
               action_index, object_index] =    ((1 - ALPHA)*q_func[current_state_1, current_state_2, action_index, object_index] +
                                                  ALPHA*(reward + GAMMA*np.max(q_func[next_state_1, next_state_2, :, :])))
    else:
        q_func[current_state_1, current_state_2,
               action_index, object_index] = (
                    (1 - ALPHA) * q_func[current_state_1, current_state_2, action_index, object_index] +
                    ALPHA * reward)
    return None  # This function shouldn't return anything


def run_episode(is_for_training: bool):
    """ Runs one episode
    If for training, update Q function
    If for testing, computes and return cumulative discounted reward

    q_func is freaking global

    Args:
        is_for_training (bool): True if for training

    Returns:
        None
    """
    epsilon = TRAINING_EP if is_for_training else TESTING_EP

    epi_reward = 0.
    t = 0
    # initialize for each episode
    # q_func = ...

    (current_room_desc, current_quest_desc, terminal) = framework.newGame()
    current_room_index = dict_room_desc[current_room_desc]
    current_quest_index = dict_quest_desc[current_quest_desc]

    while not terminal:
        # Choose next action and execute
        action_index, object_index = epsilon_greedy(current_room_index, current_quest_index,
                                                    q_func, epsilon)
        next_room_desc, next_quest_desc, reward, terminal = framework.step_game(
            current_room_desc, current_quest_desc, action_index, object_index
        )
        next_room_index = dict_room_desc[next_room_desc]
        next_quest_index = dict_quest_desc[next_quest_desc]

        if is_for_training:
            # update Q-function.
            tabular_q_learning(q_func,
                               current_room_index, current_quest_index,
                               action_index, object_index,
                               reward,
                               next_room_index, next_quest_index,
                               terminal)
        else:
        # if not is_for_training:
            # update reward
            epi_reward += (GAMMA**t)*reward

        # prepare next step
        t += 1

    if not is_for_training:
        return epi_reward


def run_epoch():
    """Runs one epoch and returns reward averaged over test episodes"""
    rewards = []

    for _ in range(NUM_EPIS_TRAIN):
        run_episode(is_for_training=True)

    for _ in range(NUM_EPIS_TEST):
        rewards.append(run_episode(is_for_training=False))

    return np.mean(np.array(rewards))


def run():
    """Returns array of test reward per epoch for one run"""
    global q_func
    q_func = np.zeros((NUM_ROOM_DESC, NUM_QUESTS, NUM_ACTIONS, NUM_OBJECTS))

    single_run_epoch_rewards_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=80)
    for _ in pbar:
        single_run_epoch_rewards_test.append(run_epoch())
        pbar.set_description(
            "Avg reward: {:0.6f} | Ewma reward: {:0.6f}".format(
                np.mean(single_run_epoch_rewards_test),
                utils.ewma(single_run_epoch_rewards_test)))
    return single_run_epoch_rewards_test


if __name__ == '__main__':
    # Data loading and build the dictionaries that use unique index for each state
    (dict_room_desc, dict_quest_desc) = framework.make_all_states_index()
    NUM_ROOM_DESC = len(dict_room_desc)
    NUM_QUESTS = len(dict_quest_desc)

    # set up the game
    framework.load_game_data()

    epoch_rewards_test = []  # shape NUM_RUNS * NUM_EPOCHS

    for _ in range(NUM_RUNS):
        epoch_rewards_test.append(run())

    epoch_rewards_test = np.array(epoch_rewards_test)

    x = np.arange(NUM_EPOCHS)
    fig, axis = plt.subplots()
    axis.plot(x, np.mean(epoch_rewards_test,
                         axis=0))  # plot reward per epoch averaged per run
    axis.set_xlabel('Epochs')
    axis.set_ylabel('reward')
    axis.set_title(('Tablular: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    plt.show()
