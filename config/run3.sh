ex=3
python3 -m main \
--hyperparameter config/hyperparameters${ex}.yaml \
--device cuda \
--mode rgb_array \
--train &> logs/${ex}.log
