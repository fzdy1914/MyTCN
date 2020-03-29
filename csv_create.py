import json

with open("./segment.json", 'r') as load_f:
    load_dict = json.load(load_f)

with open("./ans.csv", 'w') as f:
    f.write("Id,Category\n")

    for i in range(len(load_dict)):
        f.write(str(i) + "," + str(load_dict[i]) + "\n")

