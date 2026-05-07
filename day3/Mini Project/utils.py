import hashlib
import re

def encrypt_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def validate_password(password):
    if len(password) < 6:
        raise InvalidPasswordError("Password must be at least 6 characters")
    if not re.search(r"[A-Z]", password):
        raise InvalidPasswordError("Must contain uppercase letter")
    if not re.search(r"[0-9]", password):
        raise InvalidPasswordError("Must contain a number")
    return True