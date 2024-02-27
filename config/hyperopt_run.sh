export branch_name=$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
echo "Running on branch ${branch_name}"
screen -S "${branch_name}_$(date +%s%3N)" bash -c "bash config/hyperopt_run_screen.sh"
