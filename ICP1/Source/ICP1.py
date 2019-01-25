'''
ICP 1 CS 5590 UMKC
Kenton Hanifl
'''

Name = input("Please input your first and last name")
print("Reversed:",Name[::-1])

#assumes user will enter an integer
Num1 = input("Please input a number.")
Num2 = input("Please input a second number.")
Sum = int(Num1) + int(Num2)
print("The sum of the numbers is:",str(Sum))

Sentence = input("Write any sentence and the letters and digits will be counted.")
Digits = 0
Letters = 0

for char in Sentence:
    if char.isdigit():
        Digits += 1
    else:
        Letters += 1

print("Letters: {0} Digits {1}.".format(str(Letters),str(Digits)))
