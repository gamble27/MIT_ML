from typing import Dict, List, Any
import unittest


class MarkovModel:
    def __init__(self):
        self.states = set()
        self.actions = {}
        self.transitions = {}
        self.rewards = {}
        self.v_values = {}
        self.q_values = {}

    def q_value_iteration_step(self, curr_q) -> Dict[Any, Dict[Any, float]]:
        q_values = {}
        for state in self.states:
            q_values[state] = {}
            for action in self.actions[state]:
                q_values[state][action] = 


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
