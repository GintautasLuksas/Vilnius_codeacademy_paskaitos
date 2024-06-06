#9. Sukurkite klasę „ReservationSystem“, kuri tvarko paslaugos
# (pvz., viešbučio kambarių, restorano stalų) rezervacijas.
# Įdiekite rezervacijų kūrimo, rezervacijų atšaukimo, prieinamumo tikrinimo
# ir visų rezervacijų peržiūros metodus.

class ReservationSystem:
    def __init__(self, max_rooms, max_tables):
        self.max_rooms = max_rooms
        self.max_tables = max_tables
        self.rooms_reserved = 0
        self.tables_reserved = 0
        self.room_reservations = {}
        self.table_reservations = {}

    def room_reservation(self, guest_name):
        if self.rooms_reserved < self.max_rooms:
            self.rooms_reserved += 1
            self.room_reservations[self.rooms_reserved] = guest_name
            print(f"Room reserved for {guest_name}")
        else:
            print('No more rooms available')

    def table_reservation(self, guest_name):
        if self.tables_reserved < self.max_tables:
            self.tables_reserved += 1
            self.table_reservations[self.tables_reserved] = guest_name
            print(f"Table reserved for {guest_name}")
        else:
            print('No more tables available')

    def available_reservations(self):
        print(f"Available rooms: {self.max_rooms - self.rooms_reserved}")
        print(f"Available tables: {self.max_tables - self.tables_reserved}")

    def show_reservations(self):
        print("Room reservations:")
        for room_num, guest_name in self.room_reservations.items():
            print(f"Room {room_num}: {guest_name}")

        print("\nTable reservations:")
        for table_num, guest_name in self.table_reservations.items():
            print(f"Table {table_num}: {guest_name}")


reservation_system = ReservationSystem(6, 20)
reservation_system.room_reservation("Gintas")
reservation_system.room_reservation("Bronius")
reservation_system.table_reservation("Jolita")
reservation_system.available_reservations()
reservation_system.show_reservations()


