def entry_check(entry):
    if not isinstance(entry, str):
        raise TypeError("Įvestis turi būti teksto tipo.")
    if len(entry) < 10:
        raise ValueError("Įvestis turi turėti bent 10 simbolių.")
    if not entry.isalnum():
        raise ValueError("Įvestis turi būti sudaryta tik iš raidžių ir skaičių.")
    return True
import pytest

@pytest.fixture
def valid_entry():
    return "TeisingaĮvestis1"

@pytest.fixture
def short_entry():
    return "Trumpas"

@pytest.fixture
def non_string_entry():
    return 12345

def test_valid_entry(valid_entry):
    assert entry_check(valid_entry) == True

def test_short_entry(short_entry):
    with pytest.raises(ValueError, match="Įvestis turi turėti bent 10 simbolių."):
        entry_check(short_entry)

def test_non_string_entry(non_string_entry):
    with pytest.raises(TypeError, match="Įvestis turi būti teksto tipo."):
        entry_check(non_string_entry)
