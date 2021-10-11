import math
import numpy as np
import string
digs = string.digits + string.ascii_letters

def int2base(x, base):
    if x < 0:
        sign = -1
    elif x == 0:
        return digs[0]
    else:
        sign = 1

    x *= sign
    digits = []

    while x:
        digits.append(digs[int(x % base)])
        x = int(x / base)

    if sign < 0:
        digits.append('-')

    digits.reverse()

    return ''.join(digits)

def to_arrangement(decimal_msg,nb_classes, nb_classes_per_image):

    nb_combinations = math.factorial(nb_classes)/ math.factorial(nb_classes-nb_classes_per_image)
    encoded = ""
    #encoded = int2base(decimal_msg, nb_combinations)

    return encoded, nb_combinations


print(to_arrangement(1846548,10, 9))

max_nb = 10
vals = []
for i in range(max_nb):
    for j in range(max_nb):
        for k in range(max_nb):
            vals.append(math.pow(i,j)+k)

vals = np.array(vals)
print(vals.shape)
