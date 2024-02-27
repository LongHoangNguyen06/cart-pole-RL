branch_name="$(git symbolic-ref HEAD 2>/dev/null)" ||
python3 -m main \
--device cuda \
--mode rgb_array \
--train &> logs/${branch_name}.log
