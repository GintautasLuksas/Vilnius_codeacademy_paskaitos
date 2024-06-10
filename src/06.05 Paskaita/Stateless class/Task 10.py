class FileUtils:
    @staticmethod
    def read():
        with open('test.txt', 'r') as file:
            print(file.read())

    @staticmethod
    def write(text):
        with open('test.txt', 'w') as file:
            written = file.write(text)
        print(written)

    @staticmethod
    def append(text):
        with open('test.txt', 'a') as file:
            written = file.write(text)
        print(written)


mylife = FileUtils()
mylife.write('Jonukas ir grytute\n')
mylife.read()
mylife.append('Pridedu papildomai\n')
mylife.read()