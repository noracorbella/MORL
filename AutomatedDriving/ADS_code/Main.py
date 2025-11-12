import numpy as np
from ADS_Environment import Environment
from value_iteration_algorithm_original import value_iteration
from q_learning_algorithm import q_learning
from auxiliary_functions import tester

Q_LEARNING = 0
VALUE_ITERATION = 1

if __name__ == "__main__":
    Training = True  # Set to True to apply algorithm. Set to false to directly observed learned policy.
    weights = [1.0, 100.0, 100.0]
    algorithm_used = VALUE_ITERATION

    train_policy_name = "value_iteration_policy.npy"  # "train_policy.npy"
    test_policy_name = "value_iteration_policy.npy"

    if Training:
        env = Environment(weights=weights)
        save = True

        # Training
        if algorithm_used == Q_LEARNING:
            policy, q = q_learning(env, alpha=0.8, gamma=0.7)
        elif algorithm_used == VALUE_ITERATION:
            policy, q = value_iteration(env, discount_factor=0.7)

        if save:
            np.save(train_policy_name, policy)  #
        print("-------------------")
        print("Finnished!!!")

    if train_policy_name != test_policy_name:
        print(
            "WARNING: Notice that the policy that will appear now is not the policy previously trained!!"
        )
    policy = np.load(test_policy_name)
    env = Environment(weights=weights)
    tester(env, policy, drawing=True)
