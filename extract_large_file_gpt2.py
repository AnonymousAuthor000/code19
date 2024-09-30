import re
import struct

# Read the content of the C++ file
with open('tensorflow-2.9.1/tensorflow/lite/examples/coder/ocgjzw.cc', 'r') as file:
    content = file.read()

# Extract the "filter_raw" data using regular expression
pattern = re.compile(r'float\s+filter_raw\[\d+\]\s*=\s*\{([^}]*)\};', re.DOTALL)
match = pattern.search(content)

if match:
    data_str = match.group(1)
    # Convert the data string to a list of floats
    data_list = [float(x) for x in data_str.split(',')]

    # Write the data to a binary file
    with open('tensorflow-2.9.1/tensorflow/lite/examples/coder/data.bin', 'wb') as bin_file:
        bin_file.write(struct.pack(f'{len(data_list)}f', *data_list))

    print("Data has been successfully extracted to data.bin")
else:
    print("No matching data found in the file.")
