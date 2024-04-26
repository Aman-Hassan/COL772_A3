def clean_file(file_name):
    # Open the file in read mode and read lines
    with open(file_name, 'r') as file:
        lines = file.readlines()

    # Remove leading spaces
    lines = [line.lstrip() for line in lines]

    # Ensure only 142 lines are present
    lines = lines[:142]

    # Add a newline at the end if not present
    if lines[-1][-1] != '\n':
        lines[-1] += '\n'

    # Open the file in write mode and overwrite
    with open(file_name, 'w') as file:
        file.writelines(lines)

def remove_non_ascii(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        contents = f.read()

    # Remove non-ASCII characters
    cleaned_contents = contents.encode("ascii", "ignore").decode()

    # Write the cleaned contents back to the file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(cleaned_contents)

import os
import sys
# Clean the file
clean_file(os.path.join(sys.argv[1], 'elife.txt'))
clean_file(os.path.join(sys.argv[1], 'plos.txt'))
remove_non_ascii(os.path.join(sys.argv[1], 'elife.txt'))
remove_non_ascii(os.path.join(sys.argv[1], 'plos.txt'))
