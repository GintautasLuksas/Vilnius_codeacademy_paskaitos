#3. Sukurkite klasę Timer, kuri gali paleisti, sustabdyti
# ir iš naujo nustatyti laikmatį bei nurodyti praėjusį laiką sekundėmis. Naudokite time python paketą.

import time

class Timer:
    start_time = None

    @staticmethod
    def paleisti():
        Timer.start_time = time.time()

    @staticmethod
    def sustabdyti():
        return time.time() - Timer.start_time

    @staticmethod
    def reset():
        Timer.start_time = None


Timer.paleisti()
time.sleep(5)
elapsed_time = Timer.sustabdyti()
print("Praėjo laikas:", elapsed_time, "sekundės")

Timer.reset()
