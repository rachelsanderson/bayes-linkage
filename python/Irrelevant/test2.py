import json

with open('distros.json', 'r') as f:
    distros_dict = json.load(f)

for distro in distros_dict:
    print(distro['Name'])
