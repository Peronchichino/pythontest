import random

while True:
    try:
        num = int(input("Enter number: "))
        
        randomNum = random.randint(1, 100)
        
        if(num == randomNum):
            print("Congrats, guessed the number correct")
        else:
            if(randomNum > num):
                print("Number was greater than what you entered")
            else:
                print("Number was less than what u entered")
            print("Try again")
    except ValueError:
        print("Invalid input")