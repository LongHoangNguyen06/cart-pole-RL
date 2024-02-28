#!/bin/bash

# Define the directories you want to sync
directories=("config" "images" ".vscode")

# Define the files you want to sync
files=(.gitignore *.py *.md *.txt)

# Define the list of servers
servers=("gpu-server" "gpu-server1")

# Loop through each server
for server in "${servers[@]}"
do
    # Loop through each directory
    for directory in "${directories[@]}"
    do
        # rsync the directory to the server
        rsync -a --no-owner "$directory" "$server:/home/long/Documents/GitHub/cart-pole-RL/"
    done

    # rsync the files to the server
    rsync -a --no-owner "${files[@]}" "$server:/home/long/Documents/GitHub/cart-pole-RL/"

    # rsync log
    rsync -a --no-owner "$server:/home/long/Documents/GitHub/cart-pole-RL/logs" .
done