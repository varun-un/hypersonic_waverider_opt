def remove_info_lines(input_file, output_file):
    try:
        i = 0
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                if line.strip() != "":

                    if "[ info ]" not in line and "diverged at:" not in line:
                        print(line)
                        outfile.write(line)
                        i += 1
                        if i > 1000:
                            break
        print(f"Processing complete. Filtered file saved to {output_file}.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
# Replace 'large_file.txt' with the path to your input file
# and 'filtered_file.txt' with the desired output file path.
input_file = "C:\\Users\\varun\\Downloads\\trace_pt2\\trace_pt2.log"
output_file = 'filtered_file.log'
remove_info_lines(input_file, output_file)
