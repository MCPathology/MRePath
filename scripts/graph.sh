#!/bin/bash

studys = ("blca")

for subdir in "${studys[@]}"; do
    echo "Processing $subdir..."

    h5_path="../WSIdata/$subdir/h5_files"
    graph_save_path="../WSIdata/$subdir/whole_graph_files"

    python extract_graph.py --h5_path "$h5_path" --graph_save_path "$graph_save_path"

    if [ $? -eq 0 ]; then
        echo "Successfully processed $subdir."
    else
        echo "Failed to process $subdir."
        exit 1
    fi

done

echo "All processing complete."