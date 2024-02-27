EXPERIMENT=10
python3 -m main \
--hyperparameter config/hyperparameters${EXPERIMENT}.yaml \
--device cuda \
--mode rgb_array \
--train &> logs/${EXPERIMENT}.log
