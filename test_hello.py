from hello import say_hello, add

# import subprocess
# import sys
# import os


def test_say_hello():
    assert (
        say_hello("Joseph")
        == "Hello, Joseph, welcome to Data Engineering Systems (IDS 706)!"
    )
    assert (
        say_hello("Joseph")
        == "Hello, Sam, welcome to Data Engineering Systems (IDS 706)!"
    )
    # Edge cases
    assert say_hello("") == "Hello, , welcome to Data Engineering Systems (IDS 706)!"


def test_add():
    assert add(2, 3) == 5
    # Edge cases
    assert add(1_000_000, 2_000_000) == 3_000_000
    assert add(-100, 100) == 0
