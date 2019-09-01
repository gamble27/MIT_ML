"""Linear QL agent"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import project4_text_game.framework as framework
import project4_text_game.utils as utils
import os


DEBUG = False


GAMMA = 0.5  # discounted factor
TRAINING_EP = 0.5  # epsilon-greedy parameter for training
TESTING_EP = 0.05  # epsilon-greedy parameter for testing
NUM_RUNS = 5  # 10
NUM_EPOCHS = 600
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 0.001  # learning rate for training

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)


def tuple2index(action_index, object_index):
    """Converts a tuple (a,b) to an index c"""
    return action_index * NUM_OBJECTS + object_index


def index2tuple(index):
    """Converts an index c to a tuple (a,b)"""
    return index // NUM_OBJECTS, index % NUM_OBJECTS


def epsilon_greedy(state_vector, theta, epsilon):
    """Returns an action selected by an epsilon-greedy exploration policy

    Args:
        state_vector (np.ndarray): extracted vector representation
        theta (np.ndarray): current weight matrix
        epsilon (float): the probability of choosing a random command

    Returns:
        (int, int): the indices describing the action/object to take
    """
    if np.random.choice([1, 0], p=[epsilon, 1 - epsilon]):
        action_index = np.random.randint(0, NUM_ACTIONS)
        object_index = np.random.randint(0, NUM_OBJECTS)
    else:
        action_index, object_index = 0, 0
        max_q_value = (theta @ state_vector)[tuple2index(action_index, object_index)]
        for a in range(NUM_ACTIONS):
            for o in range(NUM_OBJECTS):
                curr_q_value = (theta @ state_vector)[tuple2index(a, o)]
                if curr_q_value > max_q_value:
                    max_q_value = curr_q_value
                    action_index = a
                    object_index = o

    return action_index, object_index


def linear_q_learning(theta, current_state_vector, action_index, object_index,
                      reward, next_state_vector, terminal):
    """Update theta for a given transition

    Args:
        theta (np.ndarray): current weight matrix
        current_state_vector (np.ndarray): vector representation of current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent recieves from playing current command
        next_state_vector (np.ndarray): vector representation of next state
        terminal (bool): True if this epsiode is over

    Returns:
        None
    """
    if terminal:
        max_q = 0
    else:
        max_q = max([(theta @ next_state_vector)[tuple2index(a, o)]
                     for a in range(NUM_ACTIONS)
                     for o in range(NUM_OBJECTS)])
    q_value = (theta @ current_state_vector)[tuple2index(action_index, object_index)]
    theta[tuple2index(action_index, object_index)] += ALPHA*(
        reward + GAMMA*max_q - q_value
    ) * current_state_vector


def run_episode(is_for_training):
    """ Runs one episode
    If for training, update Q function
    If for testing, computes and return cumulative discounted reward

    theta & dictionary are freaking global

    Args:
        is_for_training (bool): True if for training

    Returns:
        None
    """
    epsilon = TRAINING_EP if is_for_training else TESTING_EP
    epi_reward = 0.
    t = 0
    # initialize for each episode
    # theta = ... # global

    (current_room_desc, current_quest_desc, terminal) = framework.newGame()
    while not terminal:
        # Choose next action and execute
        current_state = current_room_desc + current_quest_desc
        current_state_vector = utils.extract_bow_feature_vector(
            current_state,
            dictionary
        )
        action_index, object_index = epsilon_greedy(
            current_state_vector, theta, epsilon
        )
        next_room_desc, next_quest_desc, reward, terminal = framework.step_game(
            current_room_desc, current_quest_desc, action_index, object_index
        )
        next_state_vector = utils.extract_bow_feature_vector(
            next_room_desc + next_quest_desc,
            dictionary
        )

        if is_for_training:
            # update Q-function.
            linear_q_learning(
                theta,
                current_state_vector,
                action_index, object_index,
                reward,
                next_state_vector,
                terminal
            )

        if not is_for_training:
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
    global theta
    theta = np.zeros([action_dim, state_dim])

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
    state_texts = utils.load_data('game.tsv')
    dictionary = utils.bag_of_words(state_texts)
    state_dim = len(dictionary)
    action_dim = NUM_ACTIONS * NUM_OBJECTS

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
    axis.set_title(('Linear: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    os.system('spd-say "your program has finished"')
    plt.show()
    """
    Avg reward: 0.193014 | Ewma reward: 0.214379: 100%|█| 600/600 [06:04<00:00,  1.62it/s]
    Avg reward: 0.192132 | Ewma reward: 0.187196: 100%|█| 600/600 [06:09<00:00,  1.56it/s]
    Avg reward: 0.195904 | Ewma reward: 0.214363: 100%|█| 600/600 [07:00<00:00,  1.55it/s]
    Avg reward: 0.194645 | Ewma reward: 0.194678: 100%|█| 600/600 [06:05<00:00,  1.64it/s]
    Avg reward: 0.187563 | Ewma reward: 0.195722: 100%|█| 600/600 [07:26<00:00,  1.12s/it]
    """
