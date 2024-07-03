import pytest


def validate_age(age):
    if not (0 <= age <= 150):
        raise ValueError("Age must be between 0 and 150 years")


def test_validate_age_value_error():
    validate_age(30)

    with pytest.raises(ValueError):
        validate_age(-1)

    with pytest.raises(ValueError):
        validate_age(160)

    validate_age(0)
    validate_age(150)
