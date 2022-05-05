import sys
import json
import matplotlib
import matplotlib.pyplot as plt

def create_pretty_names(names):
    pretty_names = []
    for name in names:
        if name == 'tdt4265':
            pretty_name = 'Task 2.1'
        elif name == 'task2_4_min_sizes_v3':
            pretty_name = 'Task 2.4 (min_sizes and a.r.)'
        else:
            pretty_name = name.replace('_', '.')
            pretty_name = pretty_name.replace('v', '')
            pretty_name = 'T' + pretty_name[1:]
            pretty_name = pretty_name[:4] + ' ' + pretty_name[4:]
        pretty_names.append(pretty_name)
    return pretty_names

### Colors used for the different tasks ###
# tdt4265 - red
# task2_2 - blue
# task2_3_v1 - black
# task2_3_v2 - g
# task2_3_v3 - orange
# task2_3_v4 - darkmagenta

args = sys.argv[1:]
file_names = args[0::2]
colors = args[1::2]

loss_class = {}
loss_reg = {}
coco_map = {}
rider_ap = {}
bus_ap = {}
person_ap = {}

for file_name in file_names:
    loss_class[file_name] = {}
    loss_reg[file_name] = {}
    coco_map[file_name] = {}
    rider_ap[file_name] = {}
    bus_ap[file_name] = {}
    person_ap[file_name] = {}

    file = open(f'outputs/{file_name}/logs/scalars.json', 'r')

    for line in file:
        data = json.loads(line)
        if 'metrics/mAP' in data:
            coco_map[file_name][int(data['global_step'])] = float(data['metrics/mAP'])
            rider_ap[file_name][int(data['global_step'])] = float(data['metrics/AP_rider'])
            bus_ap[file_name][int(data['global_step'])] = float(data['metrics/AP_bus'])
            person_ap[file_name][int(data['global_step'])] = float(data['metrics/AP_person'])
        if 'loss/total_loss' in data:
            loss_class[file_name][int(data['global_step'])] = float(data['loss/classification_loss'])
            loss_reg[file_name][int(data['global_step'])] = float(data['loss/regression_loss'])

font = {'size' : 16}
matplotlib.rc('font', **font)

plt.title(f'Classification Loss')
for i in range(len(args)//2):
    plt.plot(list(loss_class[file_names[i]].keys()), list(loss_class[file_names[i]].values()), color=colors[i])
plt.legend(create_pretty_names(file_names))
ax = plt.gca()
ax.set_ylim([0, 10])
plt.xlabel('Number of steps')
plt.ylabel('Loss')
plt.grid()
plt.savefig('figures/graphs/loss_class_' + '-'.join(file_names) + '.png')
plt.savefig('figures/svgs/graphs/loss_class_' + '-'.join(file_names) + '.svg')
plt.clf()

plt.title(f'Regression Loss')
for i in range(len(args)//2):
    plt.plot(list(loss_reg[file_names[i]].keys()), list(loss_reg[file_names[i]].values()), color=colors[i])
plt.legend(create_pretty_names(file_names))
plt.xlabel('Number of steps')
plt.ylabel('Loss')
plt.grid()
plt.savefig('figures/graphs/loss_reg_' + '-'.join(file_names) + '.png')
plt.savefig('figures/svgs/graphs/loss_reg_' + '-'.join(file_names) + '.svg')
plt.clf()

font = {'size' : 10}
matplotlib.rc('font', **font)

plt.title(f'mAP@0.5:0.95')
for i in range(len(args)//2):
    plt.plot(list(coco_map[file_names[i]].keys()), list(coco_map[file_names[i]].values()), color=colors[i])
plt.legend(create_pretty_names(file_names))
plt.xlabel('Number of steps')
plt.ylabel('mAP')
plt.grid()
plt.savefig('figures/graphs/map_' + '-'.join(file_names) + '.png')
plt.savefig('figures/svgs/graphs/map_' + '-'.join(file_names) + '.svg')
plt.clf()

font = {'size' : 10}
matplotlib.rc('font', **font)

plt.title(f'AP - Rider')
for i in range(len(args)//2):
    plt.plot(list(rider_ap[file_names[i]].keys()), list(rider_ap[file_names[i]].values()), color=colors[i])
plt.legend(create_pretty_names(file_names))
plt.xlabel('Number of steps')
plt.ylabel('AP')
plt.grid()
plt.savefig('figures/graphs/ap_rider_' + '-'.join(file_names) + '.png')
plt.savefig('figures/svgs/graphs/ap_rider_' + '-'.join(file_names) + '.svg')
plt.clf()

font = {'size' : 16}
matplotlib.rc('font', **font)

plt.title(f'AP - Bus')
for i in range(len(args)//2):
    plt.plot(list(bus_ap[file_names[i]].keys()), list(bus_ap[file_names[i]].values()), color=colors[i])
plt.legend(create_pretty_names(file_names))
plt.xlabel('Number of steps')
plt.ylabel('AP')
plt.grid()
plt.savefig('figures/graphs/ap_bus_' + '-'.join(file_names) + '.png')
plt.savefig('figures/svgs/graphs/ap_bus_' + '-'.join(file_names) + '.svg')
plt.clf()

font = {'size' : 16}
matplotlib.rc('font', **font)

plt.title(f'AP - Person')
for i in range(len(args)//2):
    plt.plot(list(person_ap[file_names[i]].keys()), list(person_ap[file_names[i]].values()), color=colors[i])
plt.legend(create_pretty_names(file_names))
plt.xlabel('Number of steps')
plt.ylabel('AP')
plt.grid()
plt.savefig('figures/graphs/ap_person_' + '-'.join(file_names) + '.png')
plt.savefig('figures/svgs/graphs/ap_person_' + '-'.join(file_names) + '.svg')
plt.clf()
