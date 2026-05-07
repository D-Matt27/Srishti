import random
words=['python','java','computer','developer']
stages=[
    """
    -----
        |
        |
        |
        |
        |
    =========

    """,
    """
    -----
    |   |
        |
        |
        |
        |
    =========

    """,
    """
    -----
    |   |
    O   |
        |
        |
        |
    =========

    """,
    """
    -----
    |   |
    O   |
    |   |
        |
        |
    =========

    """,
    """
    -----
    |   |
    O   |
   /|   |
        |
        |
    =========

    """,
    """
    -----
    |   |
    O   |
   /|\\  |
        |
        |
    =========

    """,
    """
    -----
    |   |
    O   |
   /|\\  |
   /    |
        |
    =========

    """,
    """
    -----
    |   |
    O   |
   /|\\  |
   / \\  |
        |
    =========

    """]

word=random.choice(words)
guessed_letters=[]
errors=0
while(errors<7):
    print(stages[errors])
    
    if errors==2:
        if word=='python':
            print("HINT:It is a popular interpreted programming language")
        elif word=='java':
            print("HINT:This programming language was used to develop MineCraft")
        elif word=='computer':
            print("HINT:PC stands for Personal ________")
        else:
            print("HINT:What do you call someone that DEVELOPS something")
        
    display_word=""
    for letter in word:
        if letter in guessed_letters:
            display_word+=letter+" "
        else:
            display_word+="_ "

    print("Word:",display_word.strip())
    print("Guessed:",guessed_letters)
    print("Lives:",7-errors)

    all_guessed=True
    for letter in word:
        if letter not in guessed_letters:
            all_guessed=False
            break
    if all_guessed:
        print("Word Guessed Successfully!\nWord is:",word)
        break
    
    guess=input("Enter your guess:")
    if not guess.isalpha() or len(guess)!=1:
        print("Invalid Input\nTry Again")
        continue

    if guess in guessed_letters:
        print("Already Guessed")
        continue

    guessed_letters.append(guess)

    if guess not in word:
        errors += 1
        print("WRONG GUESS")

if errors==7:
    print(stages[7])
    print("Game Over. The word was:",word)