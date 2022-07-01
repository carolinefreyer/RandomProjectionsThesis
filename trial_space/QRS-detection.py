import numpy as np
import load_data_MITBIH as ld

with open('C:/Users/carol/OneDrive - Delft University of Technology/Documents/NetBeansProjects/QRS/beats_all.txt') as f:
    lines = f.readlines()
    beats_all = np.array([float(item.strip()) for item in lines[0][1:-2].split(",")])

with open('C:/Users/carol/OneDrive - Delft University of Technology/Documents/NetBeansProjects/QRS/events.txt') as f:
    lines = f.readlines()
    events = np.array([float(item.strip()) for item in lines[0][1:-1].split(",")])

with open('C:/Users/carol/OneDrive - Delft University of Technology/Documents/NetBeansProjects/QRS/ridges_2.txt') as f:
    lines = f.readlines()
    ridges = np.array([float(item.strip()) for item in lines[0][1:-2].split(",")])

with open('C:/Users/carol/OneDrive - Delft University of Technology/Documents/NetBeansProjects/QRS/results.txt') as f:
    lines = f.readlines()
    results = np.array([int(item.strip()) for item in lines[0][1:-2].split(",")])

diff = [events[i+1]-events[i] for i in range(len(events)-1)]
print(diff)
print(np.mean(diff))

# record, annotation = ld.load_mit_bih_data("100", 0, None)


