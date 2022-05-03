import sys
import json
import matplotlib.pyplot as plt

def create_pretty_names(names):
    pretty_names = []
    for name in names:
        if name == 'tdt4265':
            pretty_name = 'Task 2.1'
        else:
            pretty_name = name.replace('_', '.')
            pretty_name = pretty_name.replace('v', '')
            pretty_name = 'T' + pretty_name[1:]
            pretty_name = pretty_name[:4] + ' ' + pretty_name[4:]
        pretty_names.append(pretty_name)
    return pretty_names

args = sys.argv[1:]
file_names = args[0::2]
colors = args[1::2]

loss_class = {}
loss_reg = {}
coco_map = {}

for file_name in file_names:
    loss_class[file_name] = {}
    loss_reg[file_name] = {}
    coco_map[file_name] = {}

    file = open(f'outputs/{file_name}/logs/scalars.json', 'r')

    for line in file:
        data = json.loads(line)
        if 'metrics/mAP' in data:
            coco_map[file_name][data['global_step']] = data['metrics/mAP']
        if 'loss/total_loss' in data:
            loss_class[file_name][data['global_step']] = data['loss/classification_loss']
            loss_reg[file_name][data['global_step']] = data['loss/regression_loss']

plt.title(f'Classification Loss')
for i in range(len(args)//2):
    plt.plot(loss_class[file_names[i]].keys(), loss_class[file_names[i]].values(), color=colors[i])
plt.legend(create_pretty_names(file_names))
ax = plt.gca()
ax.set_ylim([0, 10])
plt.xlabel('Number of steps')
plt.ylabel('Loss')
plt.grid()
plt.savefig('scripts/loss_class_' + '-'.join(file_names) + '.svg')
plt.clf()

plt.title(f'Regression Loss')
for i in range(len(args)//2):
    plt.plot(loss_reg[file_names[i]].keys(), loss_reg[file_names[i]].values(), color=colors[i])
plt.legend(create_pretty_names(file_names))
plt.xlabel('Number of steps')
plt.ylabel('Loss')
plt.grid()
plt.savefig('scripts/loss_reg_' + '-'.join(file_names) + '.svg')
plt.clf()

plt.title(f'mAP@0.5:0.95')
for i in range(len(args)//2):
    plt.plot(coco_map[file_names[i]].keys(), coco_map[file_names[i]].values(), color=colors[i])
plt.legend(create_pretty_names(file_names))
plt.xlabel('Number of steps')
plt.ylabel('mAP')
plt.grid()
plt.savefig('scripts/map_' + '-'.join(file_names) + '.svg')
plt.clf()
