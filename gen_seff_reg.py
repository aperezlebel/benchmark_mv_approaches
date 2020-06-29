"""Generate command for seff classification tasks."""

ids_reg = [2, 5, 8, 11, 14, 17, 20, 29, 32]


while True:
    e = input('>')

    s = ' ; '.join([f'seff {e}_{id}' for id in ids_reg])
    print(s)
