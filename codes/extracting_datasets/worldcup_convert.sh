#!/bin/bash

# Loop through all files in the "datasets" folder
for filename in /Users/roozbeh/Documents/benchmark_code/files/worldcup_access_log/*; do
    # Extract the base name without the directory and extension
    base_name=$(basename "$filename")
    echo $base_name
    # Remove the ".gz" extension if present
    base_name=${base_name%.gz}
    echo $base_name
    # Execute the command for each file
    gzip -dc "/Users/roozbeh/Documents/benchmark_code/files/worldcup/$base_name" | /Users/roozbeh/Downloads/ita_public_tools/bin/recreate /Users/roozbeh/Downloads/ita_public_tools/state/object_mappings.sort > "$base_name.out"
done   