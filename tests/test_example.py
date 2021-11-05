import pytest


def test_example():
    a = 2
    assert a == 2
    assert a != 3
    with pytest.raises(ValueError) as e:
        a = 2 / 0
    assert (
        str(e.value)
        == "Operator '-' have an undefined behavior between a matrix and a real/complex number."
    )
