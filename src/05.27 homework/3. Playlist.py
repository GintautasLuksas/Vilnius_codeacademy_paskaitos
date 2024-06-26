class Playlist:
    def __init__(self, songs: list):
        self.songs = songs

    def add_song(self, addition: str):
        self.songs.append(addition)

    def remove_song(self, removal: str):
        if removal in self.songs:
            self.songs.remove(removal)

    def list_songs(self):
        return self.songs


my_list = Playlist(['Johnydead', 'CAT', 'Cooldaddy'])
print(my_list.list_songs())
my_list.add_song('Torpeda')
print(my_list.list_songs())
my_list.remove_song('CAT')
print(my_list.list_songs())
