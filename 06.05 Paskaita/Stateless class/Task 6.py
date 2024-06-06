#6. Sukurkite klasę UserSession, kuri valdo vartotojo prisijungimo būseną. Joje turėtų būti būdai prisijungti,
# atsijungti, patikrinti, ar vartotojas yra prisijungęs, ir gauti prisijungusio vartotojo duomenis.

class UserSession:
    def __init__(self, user: str):
        self.user = user
        self.is_connected = False

    def connect(self):
        self.is_connected = True
        return 'Connected Successfully'

    def disconnect(self):
        self.is_connected = False
        return 'Disconnected Successfully'

    def is_connected(self):
        return self.is_connected

    def show_user_info(self):
        return f"User: {self.user}, Connected: {self.is_connected}"


def main_menu():
    user_session = None

    while True:
        print('''
        Welcome to our User session:
        Choose option:
        1. Connect
        2. Disconnect
        3. Check Connection Status
        4. Show User Info
        5. Exit
        ''')
        choice = input("Enter your choice: ")

        if choice == "1":
            if not user_session:
                user_name = input("Enter username: ")
                user_session = UserSession(user_name)
            print(user_session.connect())

        elif choice == "2":
            if user_session:
                print(user_session.disconnect())
                user_session = None
            else:
                print("No user connected.")

        elif choice == "3":
            if user_session:
                print("User is connected.")
            else:
                print("User is not connected.")

        elif choice == "4":
            if user_session:
                print(user_session.show_user_info())
            else:
                print("No user connected.")

        elif choice == "5":
            print("Exiting program...")
            break

        else:
            print("Invalid choice. Please enter a valid option.")

main_menu()
