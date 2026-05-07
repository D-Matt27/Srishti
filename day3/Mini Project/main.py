from auth import register, login, update_user, delete_user
from file_handler import backup_users
from exceptions import *


def dashboard(username):
    while True:
        print("\n1. Update Password")
        print("2. Delete Account")
        print("3. Logout")

        choice = input("Choose: ")

        try:
            if choice == "1":
                update_user(username)
                print("Updated successfully")

            elif choice == "2":
                delete_user(username)
                print("Deleted successfully")
                break

            elif choice == "3":
                print("Logged out")
                break

            else:
                print("Invalid choice")

        except Exception as e:
            print(e)


def main():
    while True:
        print("\n1. Register")
        print("2. Login")
        print("3. Backup Data")
        print("4. Exit")

        choice = input("Enter choice: ")

        try:
            if choice == "1":
                register()

            elif choice == "2":
                user = login()
                if user:
                    dashboard(user)

            elif choice == "3":
                backup_users()
                print("Backup created")

            elif choice == "4":
                break

            else:
                print("Invalid choice")

        except UserExistsError as e:
            print(e)
        except InvalidPasswordError as e:
            print(e)
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()


