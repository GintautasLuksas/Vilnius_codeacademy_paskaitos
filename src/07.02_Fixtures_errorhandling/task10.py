import pytest

def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Failas '{file_path}' nerastas.") from e

@pytest.fixture
def temp_file(tmp_path):
    file_path = tmp_path / "test_file.txt"
    content = "Šis yra testinis failas."
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return file_path

def test_read_existing_file(temp_file):
    content = read_file(temp_file)
    assert content == "Šis yra testinis failas."

def test_read_non_existing_file():
    with pytest.raises(FileNotFoundError) as excinfo:
        read_file("neegzistuojantis_failas.txt")
    assert "Failas 'neegzistuojantis_failas.txt' nerastas." in str(excinfo.value)

if __name__ == "__main__":
    pytest.main()
