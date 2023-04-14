
import statistics
import numpy as np
import geopy.distance
import os
import shutil

dest = "TMD_Datasets/"
src = "TMD Datalog/"
directory = os.scandir(src)

# create a new destination folder
try:
    shutil.rmtree(dest)
except:
    pass
os.mkdir(dest) 


for csv in directory:
    
    # Read and store the information in the datalog files
    if (csv.name.split('.')[1] != "csv"): continue
    
    print(csv.name)
    datalog = np.loadtxt(src + csv.name , dtype='str' , delimiter=",")

    lat_string = datalog [: , 0]
    lon_string = datalog [: , 1]
    times = datalog [: , 6]
    modes = datalog [: , 7]

    # Below new features are created and stored into an array
    time_integers = []
    for items in times:
        h, m, s = items.split(':')
        time_integers.append(int(int(h) * 3600 + int(m) * 60 + float(s)))

    delta_times = []
    distances = []
    speeds = []
    accelerations = []
    i = 0

    while i < len(time_integers) - 1:
        coords_1 = (lat_string[i], lon_string[i])
        coords_2 = (lat_string[i+1], lon_string[i+1])
        
        distance = geopy.distance.geodesic(coords_1, coords_2).km
        distances.append(distance)
            
        delta_time = time_integers[i+1] - time_integers[i]
        delta_times.append(delta_time)
        
        try:
            speed = distance / (delta_time/3600)
        except:
            print(lat_string[i], lon_string[i], distance, delta_time)
        speeds.append(speed)
        
        i += 1

    x = 0

    while x < len(speeds) - 1:
        acceleration = (speeds[x+1]-speeds[x]) / (delta_times[x]/3600)
        accelerations.append(acceleration)
        x += 1
    avg_speeds = []
    std_speeds = []
    max_speeds = []
    percentile_75_speeds = []
    percentile_50_speeds = []
    window_size = 20 # Window size used for calculating some of the features. Can be changed to make different datasets if needed
    iteration_speeds = 0

    while iteration_speeds < len(speeds)-1:
        
        window = speeds[iteration_speeds: iteration_speeds + window_size]
        
        avg_speed = sum(window) /len(window)
        std_speed = statistics.stdev(window)
        max_speed = max(window)
        percentile_75_speed = np.percentile(window, 75)
        percentile_50_speed = np.percentile(window, 50)
        
        avg_speeds.append(avg_speed)
        std_speeds.append(std_speed)
        max_speeds.append(max_speed)
        percentile_75_speeds.append(percentile_75_speed)
        percentile_50_speeds.append(percentile_50_speed)
        
        iteration_speeds+=1

    avg_accelerations = []
    std_accelerations = []
    iteration_accs = 0

    while iteration_accs < len(accelerations) - 1:
        
        window = accelerations[iteration_accs: iteration_accs + window_size]
        
        avg_acceleration = sum(window) /len(window)
        std_acceleration = statistics.stdev(window)
        
        avg_accelerations.append(avg_acceleration)
        std_accelerations.append(std_acceleration)

        
        iteration_accs+=1

    distances_opt = distances[:len(accelerations) - 1]
    speeds_opt = speeds[:len(accelerations) - 1]
    accelerations_opt = accelerations[:len(accelerations) - 1]
    avg_speeds_opt = avg_speeds[:len(accelerations) - 1]
    std_speeds_opt = std_speeds[:len(accelerations) - 1]
    max_speeds_opt = max_speeds[:len(accelerations) - 1]
    percentile_75_speeds_opt = percentile_75_speeds[:len(accelerations) - 1]
    percentile_50_speeds_opt = percentile_50_speeds[:len(accelerations) - 1]
    modes_opt = modes[:len(accelerations) - 1]

    # Save the features for each datalog file into another dataset_NAME.csv file
    entire_array = np.c_[(distances_opt, speeds_opt, accelerations_opt, avg_speeds_opt, std_speeds_opt, max_speeds_opt, percentile_75_speeds_opt, percentile_50_speeds_opt, avg_accelerations, std_accelerations, modes_opt)]
    np.savetxt(dest +  "dataset_" + csv.name, entire_array, delimiter=",", fmt='%s')


# Merge all of the created csv files into one csv file called "dataset.csv" which will be used for the machine learning training
file_list = os.scandir(dest)
combined = 'dataset.csv'
with open(dest+combined, 'a') as dset:
    for file in file_list:
        if file.name == combined: continue
        data = open(dest+file.name, 'r')
        for line in data:
            dset.write(line)
