import random

def test_random_between_zero_and_one():
    value = random.random()
    assert 0.0 <= value <= 1.0