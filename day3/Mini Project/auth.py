from file_handler import load_users, save_users
from utils import encrypt_password, validate_password
from exceptions import UserExistsError, UserNotFoundError, InvalidPasswordError
from log import write_log


def register():
    users = load_users()
    username = input("Enter username: ")

    if username in users:
        raise UserExistsError("User already exists")

    password = input("Enter password: ")
    validate_password(password)

    users[username] = encrypt_password(password)
    save_users(users)
    write_log(f"User registered: {username}")
    print("Registration successful")


def login():
    users = load_users()

    for _ in range(3):
        username = input("Username: ")
        password = input("Password: ")

        if username in users and users[username] == encrypt_password(password):
            write_log(f"User logged in: {username}")
            print("Login successful")
            return username
        else:
            print("Invalid credentials")

    print("Too many attempts")
    return None


def update_user(username):
    users = load_users()
    password = input("Enter new password: ")
    validate_password(password)

    users[username] = encrypt_password(password)
    save_users(users)
    write_log(f"User updated: {username}")


def delete_user(username):
    users = load_users()

    if username not in users:
        raise UserNotFoundError("User not found")

    del users[username]
    save_users(users)
    write_log(f"User deleted: {username}")