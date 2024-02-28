branch_name=$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
export SESSION_NAME="${branch_name}_$(date +%s%3N)"
echo "Running on branch ${branch_name}"
screen -S ${SESSION_NAME} bash -c "bash config/hyperopt_run_screen.sh"
