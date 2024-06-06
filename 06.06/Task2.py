#2.1 Įdiekite „zipfile“ paketą (yra „Python“).
#2.2 Sukurkite klasę pavadinimu FileCompressor.
#2.3 Implementuokite metodą compress(files, output_zip), kad
# suspaustumėte failų sąrašą į ZIP failą.
#2.4 Implementuokite metodą decompress(zip_file, output_dir),
# kad ištrauktumėte ZIP failo turinį.

import zipfile
class FileCompressor:
    def compress(self, files, output_zip):
