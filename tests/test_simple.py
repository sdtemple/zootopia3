import random
import unittest

def test_random_between_zero_and_one():
    value = random.random()
    assert 0.0 <= value <= 1.0

if __name__ == '__main__':
    test_random_between_zero_and_one()