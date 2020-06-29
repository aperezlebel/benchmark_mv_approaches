"""Generate command for seff classification tasks."""

ids_classif = [0, 3, 6, 9, 12, 15, 18, 27, 30]


while True:
    e = input('>')

    s = ' ; '.join([f'seff {e}_{id}' for id in ids_classif])
    print(s)
