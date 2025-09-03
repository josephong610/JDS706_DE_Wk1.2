from hello import say_hello, multiply


def test_say_hello():
    assert (
        say_hello("Joseph")
        == "Hello, Joseph, welcome to Data Engineering Systems (IDS 706)!"
    )
    # Edge case: empty string
    assert say_hello("") == "Hello, , welcome to Data Engineering Systems (IDS 706)!"


def test_multiply():
    assert multiply(2, 3) == 6
    # Edge cases
    assert multiply(1_000, 2_000) == 2_000_000
    assert multiply(-5, 5) == -25
    assert multiply(0, 100) == 0
