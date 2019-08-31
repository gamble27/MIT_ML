from typing import Dict, List, Any

import numpy as np
import unittest


class MarkovModel:
    states:         List[Any]
    actions:        Dict[Any, List[Any]]
    transitions:    List[List[float]]
    rewards:        List[List[float]]
    _init_values:   List[float]
    _init_q_values: Dict[Any, Dict[Any, float]]

    def __init__(self):
        self.states = []
        self.actions = {}  # actions[state] = [a0, a1, ...], a[i] - state.

        self.transitions = []  # a 2d array
        self.rewards = []  # same

        self._init_values = []  # 1d array
        self._init_q_values = {}

    def value_iteration_step(self, current_values: np.ndarray) -> np.ndarray:
        return np.array([])

    def q_value_iteration_step(self, current_q_values: Dict[Any, List[float]]) -> Dict[Any, List[float]]:
        q_values = {}
        for state in current_q_values:
            q_values[state] = {}
            for state2 in self.actions[state]:
                q_values[state][state2] =


        return {}

    def value_iteration(self, assign_values: bool = True) -> list:
        previous_values = np.array(self._init_values[:])
        while True:
            current_values = self.value_iteration_step(previous_values)
            if np.linalg.norm(previous_values - current_values) <= 1e-6:
                if assign_values:
                    self._init_values = list(current_values)
                return list(current_values)
            else:
                previous_values = current_values[:]

    def q_value_iteration(self, assign_q_values: bool = True) -> list:
        previous_q_values = np.array(self._init_q_values[:])
        while True:
            current_q_values = self.q_value_iteration_step(previous_q_values)
            if np.linalg.norm(previous_q_values - current_q_values) <= 1e-6:
                if assign_q_values:
                    self._init_q_values = list(current_q_values)
                return list(current_q_values)
            else:
                previous_q_values = current_q_values

    def compute_q_values(self, values: list = None, assign_q_values: bool = True) -> dict:
        """
        Computes optimal Q values
        for given V values

        :param values:          list of [optimal] V values
        :param assign_q_values: [flag] whether
                                to assign computed Q values
                                to the class variable

        :return:                computed Q values as dictionary:
                                Qs[s] = [Q(s,a[0]), ...]
        """
        if values is None:
            values = self._init_values

        q_values = {}
        for state in self.states:
            q_values[state] = []
            for action in self.actions[state]:
                ...  # todo

        if assign_q_values:
            self._init_q_values = q_values

        return {}


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
