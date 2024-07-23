import pytest


class InvalidCredentialsError(Exception):
    pass


class AccountLockedError(Exception):
    pass


def authorization(user_id, password, attempts):
    locked_users = {'Blokuotas': 'Paskyra užblokuota'}

    if user_id in locked_users:
        raise AccountLockedError(locked_users[user_id])

    if user_id == 'Morka' and password == 'Grauziu':
        return True
    else:
        if attempts >= 3:
            raise AccountLockedError("Per daug neteisingų bandymų. Paskyra užblokuota.")
        raise InvalidCredentialsError("Neteisingi prisijungimo duomenys.")


@pytest.fixture
def valid_credentials():
    return {'user_id': 'Morka', 'password': 'Grauziu'}


@pytest.fixture
def invalid_credentials():
    return {'user_id': 'Morka', 'password': 'Netinkamas'}


@pytest.fixture
def locked_user_credentials():
    return {'user_id': 'Blokuotas', 'password': 'Bet koks'}


def test_successful_authorization(valid_credentials):
    result = authorization(valid_credentials['user_id'], valid_credentials['password'], 0)
    assert result == True


def test_invalid_password(invalid_credentials):
    with pytest.raises(InvalidCredentialsError) as excinfo:
        authorization(invalid_credentials['user_id'], invalid_credentials['password'], 1)
    assert str(excinfo.value) == "Neteisingi prisijungimo duomenys."


def test_account_locked_after_three_attempts(invalid_credentials):
    with pytest.raises(AccountLockedError) as excinfo:
        authorization(invalid_credentials['user_id'], invalid_credentials['password'], 4)
    assert str(excinfo.value) == "Per daug neteisingų bandymų. Paskyra užblokuota."


def test_locked_user(locked_user_credentials):
    with pytest.raises(AccountLockedError) as excinfo:
        authorization(locked_user_credentials['user_id'], locked_user_credentials['password'], 0)
    assert str(excinfo.value) == "Paskyra užblokuota"