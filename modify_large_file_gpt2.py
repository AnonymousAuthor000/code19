import re

# Read the content of the C++ file
with open('tensorflow-2.9.1/tensorflow/lite/examples/coder/ocgjzw.cc', 'r') as file:
    content = file.read()

# Remove the line containing "float filter_raw[number_of_datapoints]={...}"
pattern = re.compile(r'float\s+filter_raw\[\d+\]\s*=\s*\{[^}]*\};', re.DOTALL)
modified_content = pattern.sub('', content)

# Write the modified content back to a new file
with open('tensorflow-2.9.1/tensorflow/lite/examples/coder/ocgjzw.cc', 'w') as file:
    file.write(modified_content)

print("The line containing 'float filter_raw[number_of_datapoints]={...}' has been successfully removed.")
