import sys
import json
import matplotlib.pyplot as plt

file_name = 'task2_3_v4'
learning_rate = {}

file = open(f'outputs/{file_name}/logs/scalars.json', 'r')

for line in file:
    data = json.loads(line)
    if 'stats/learning_rate' in data:
        learning_rate[data['global_step']] = data['stats/learning_rate']

print(learning_rate)

plt.title(f'Learning Rate')
plt.plot(learning_rate.keys(), learning_rate.values())
plt.xlabel('Number of steps')
plt.ylabel('lr')
plt.grid()
plt.savefig('scripts/lr-' + file_name + '.png')
