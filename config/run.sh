branch_name="$(git symbolic-ref HEAD 2>/dev/null)" ||

screen -S ${branch_name} "python3 -m main --device cuda --mode rgb_array --train &> logs/${branch_name}.log"
