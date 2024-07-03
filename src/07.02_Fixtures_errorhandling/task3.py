import pytest
import os
import tempfile


def word_count_in_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return len(content.split())

@pytest.fixture
def file_with_text():
    text = "Another test file with several words."
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
        temp_file.write(text)
        temp_file_path = temp_file.name
    yield temp_file_path
    os.remove(temp_file_path)

def test_word_count_in_file(file_with_text):
    assert word_count_in_file(file_with_text) == 6



