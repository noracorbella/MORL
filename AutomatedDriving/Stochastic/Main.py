import numpy as np
from ADS_Environment import Environment
from VI_Stochastic_SO import value_iteration
from q_learning_algorithm import q_learning
from CH_VI_stochastic_v2 import convexhull_VI, extract_policy_for_weights
from auxiliary_functions import tester

Q_LEARNING = 0
VALUE_ITERATION = 1
CONVEX_HULL_VI = 2 

if __name__ == "__main__":
    algorithm_used = CONVEX_HULL_VI
  

    # For CONVEX_HULL_VI
    Calculate_hulls = True # True: compute Q-hulls, False: extract policy from existing hulls

    # For Q_LEARNING and VALUE_ITERATION
    Training = True        # True: train and save policy, False: load and test existing policy

    weights = [100, 0, 10]

    train_policy_name = f"policies/VI_stochastic_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"
    test_policy_name = f"policies/VI_stochastic_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"
    v_table_name = f"policies/V_table_{weights[0]}-{weights[1]}-{weights[2]}.pkl"
    hulls_name = "policies/CH_qhulls.npy"
    chvi_policy_name = f"policies/CHVI_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy"
    
    if algorithm_used == CONVEX_HULL_VI:
        if Calculate_hulls:
            env = Environment(weights=None) 
            env.weights = weights

            q_hulls_filename = "policies/CH_VI_qhulls.pkl"
        
            q_hulls = convexhull_VI(env, theta=0.01, discount_factor=0.7, q_hulls_file=q_hulls_filename)
        
            np.save(hulls_name, q_hulls, allow_pickle=True)
            print(f"Saved Q-hulls to {hulls_name}\n")
        
        else:
            print(f"Loading Q-hulls from {hulls_name}...\n")
            q_hulls = np.load(hulls_name, allow_pickle=True).item()

            env = Environment(weights=weights)
            policy = extract_policy_for_weights(q_hulls, weights, env, env.n_actions)
            np.save(chvi_policy_name, policy)
            print(f"Saved policy to {chvi_policy_name}\n")

            print("Testing policy...\n")
            tester(env, policy, drawing=True)

    else:
        if Training:
            print(f"Training {['Q-Learning', 'Value Iteration'][algorithm_used]}\n")
            print(f"Weights: {weights}\n")

            env = Environment(weights=weights)
            env.reset()

            # ##########################################################################
            # INITIAL POSITIONS
            # ##########################################################################
            
            car_pos = env.agents[0].position
            p1_pos = env.agents[1].position
            p2_pos = env.agents[2].position

            print(f"\nInitial car position: {car_pos}")
            print(f"Initial pedestrian 1 position: {p1_pos}")
            print(f"Initial pedestrian 2 position: {p2_pos}")

            for c in env.states_agent_left:
                state_translated = env.translate_state(np.array([c, 0, 0]))
                print(state_translated)
                if list(state_translated[0]) == list(car_pos):
                    car_index = c
                    print(f"\nCar index: {car_index} translates to {car_pos}")
                    break

            for p1 in env.states_agent_right:
                state_translated = env.translate_state(np.array([0, p1, 0]))
                if list(state_translated[1]) == list(p1_pos):
                    p1_index = p1
                    print(f"P1 index: {p1_index} translates to {p1_pos}")
                    break

            for p2 in env.states_agent_right:
                state_translated = env.translate_state(np.array([0, 0, p2]))
                if list(state_translated[2]) == list(p2_pos):
                    p2_index = p2
                    print(f"P2 index: {p2_index} translates to {p2_pos}")
                    break

            state = np.array([car_index, p1_index, p2_index])
            print(f"Initial state indices: {state}\n")

            save = True

            if algorithm_used == Q_LEARNING:
                policy, q = q_learning(env, alpha=0.8, gamma=0.7)
            elif algorithm_used == VALUE_ITERATION:
                policy, q = value_iteration(env, discount_factor=0.7, v_table_file=v_table_name)

                
            if save:
                np.save(train_policy_name, policy)
                print(f"Saved policy to {train_policy_name}\n")

            print("-------------------")
            print("Finnished!!!")

            
            print("\nTesting trained policy...\n")
            tester(env, policy, drawing=True)
        
        else:
            # TESTING ONLY
            
            if train_policy_name != test_policy_name:
                print("WARNING: Notice that the policy that will appear now is not the policy previously trained!!")
            
            print(f"Loading policy from {test_policy_name}...\n")
            policy = np.load(test_policy_name)
            print("Policy loaded successfully\n")

            env = Environment(weights=weights)
            
            print("Testing policy...\n")
            tester(env, policy, drawing=True)