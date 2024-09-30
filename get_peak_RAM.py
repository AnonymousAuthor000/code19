import os
import re

def get_peak_memory_usage(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if 'mem_heap_B=' in line:
                peak_memory = int(line.split('=')[1])
            if 'mem_heap_extra_B=' in line:
                peak_memory += int(line.split('=')[1])
            if 'mem_stacks_B=' in line:
                peak_memory += int(line.split('=')[1])
    return peak_memory

# Automatically find the massif.out file in the current directory
massif_file = None
for file_name in os.listdir('.'):
    if file_name.startswith('massif.out'):
        massif_file = file_name
        break

if massif_file:
    peak_memory_usage = get_peak_memory_usage(massif_file)
    peak_memory_usage_mb = peak_memory_usage / (1024 * 1024)  # Convert bytes to megabytes
    print(f"Peak memory usage: {peak_memory_usage_mb:.2f} MB")
else:
    print("No massif.out file found in the directory.")


# massif_file = './minimal_x86_build/massif.out.738456'  # Replace <pid> with the actual process ID
# peak_memory_usage = get_peak_memory_usage(massif_file)
# print(f"Peak memory usage: {peak_memory_usage} bytes")
