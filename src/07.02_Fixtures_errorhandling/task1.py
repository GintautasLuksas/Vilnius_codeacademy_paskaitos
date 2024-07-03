import shutil
import pytest
import os


def create_file(filename: str, dir_name: str, content: str):
    filepath = f'{dir_name}/{filename}'
    with open(filepath, 'w') as f:
        f.write(content)


# Užduotis 1: Parašykite funkciją create_temp_file, kuri sukuria laikiną failą duotame kataloge. Naudokite fixtures,
# kad testams pateiktumėte laikiną katalogą, filų pavadinimus ir turinius.
# Testai: Užtikrinkite, kad failai būtų sukurti ir turinys teisingas.

@pytest.fixture
def file_data():
    dir_name = 'test_dir'
    os.mkdir(dir_name)
    file_name = 'test_file.txt'
    content = 'This is file'
    yield file_name, dir_name, content
    shutil.rmtree(dir_name)


def test_create_file(file_data):
    file_name, dir_name, content_expacted = file_data
    create_file(file_name, dir_name, content_expacted)

    file_path = f'{dir_name}/{file_name}'
    assert os.path.exists(file_path)

    with open(file_path, 'r') as f:
        content = f.read()

    assert content == content_expacted