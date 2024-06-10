#2.1 Įdiekite „zipfile“ paketą (yra „Python“).
#2.2 Sukurkite klasę pavadinimu FileCompressor.
#2.3 Implementuokite metodą compress(files, output_zip), kad
# suspaustumėte failų sąrašą į ZIP failą.
#2.4 Implementuokite metodą decompress(zip_file, output_dir),
# kad ištrauktumėte ZIP failo turinį.

import zipfile
import os

class FileCompressor:
    @staticmethod
    def compress(files, output_zip):
        with zipfile.ZipFile(output_zip, 'w') as zipf:
            for file in files:
                zipf.write(file, os.path.basename(file))

    @staticmethod
    def decompress(zip_file, output_dir):
        with zipfile.ZipFile(zip_file, 'r') as zipf:
            zipf.extractall(output_dir)

    @staticmethod
    def print_info(zip_file):
        with zipfile.ZipFile(zip_file) as file:
            print(file.infolist())

#FileCompressor.compress(["file1.txt", "file2.txt"], "compressed.zip")
#FileCompressor.decompress("compressed.zip", 'C:/Users/')
FileCompressor.print_info('compressed.zip')


