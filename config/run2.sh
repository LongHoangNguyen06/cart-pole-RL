ex=2
python3 -m main \
--architecture config/architecture${ex}.yaml \
--hyperparameter config/hyperparameters${ex}.yaml \
--device cuda \
--mode rgb_array \
--train