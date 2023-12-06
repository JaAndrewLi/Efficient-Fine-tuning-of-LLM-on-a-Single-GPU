import json
with open("alpaca_data.json", 'r') as load_file:
    load_file = json.load(load_file)
    n = 0
    a = []
    for i in load_file:
        print(len(i))
        a.append(i)
        n = n + 1
        if n == 200 :
            break
b = json.dumps(a)
f = open("alpaca_data_simplified.json", 'w')
f.write(b)
f.close()
    