my_file = 'check.txt'
with open(my_file, 'r') as f:
    s = 0
    for line in f:
        line = line.split(',')
        if len(line) != 3:
            continue
        line = line[1].split('=')[1]
        x = float(line)
        s += x
    print('Sum: ', s)