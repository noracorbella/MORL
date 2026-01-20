"""
Script de diagnòstic per comparar Value Iteration vs Lexicographic VI

Aquest script carrega ambdues policies i les compara per veure on difereixen.
"""

import numpy as np
from ADS_Environment import Environment

# Carregar policies
print("Carregant policies...")
try:
    policy_vi = np.load('../policies/VI_stochastic_1.0-100.0-100.0-policy.npy')
    print("✓ VI amb pesos [1, 100, 100] carregada")
except:
    print("✗ No es troba ../policies/VI_stochastic_1.0-100.0-100.0-policy.npy")
    policy_vi = None

try:
    policy_lex = np.load('policies/LGVI_pedestrian_priority-policy.npy')
    print("✓ Lexicographic VI (pedestrian) carregada")
except:
    print("✗ No es troba LGVI_car_priority-policy.npy")
    policy_lex = None

if policy_vi is not None and policy_lex is not None:
    # Comparar policies
    print(f"\nForma de les policies:")
    print(f"  VI:  {policy_vi.shape}")
    print(f"  Lex: {policy_lex.shape}")
    
    # Comptar diferències
    diff = (policy_vi != policy_lex)
    n_diff = np.sum(diff)
    total = policy_vi.size
    
    print(f"\nDiferències:")
    print(f"  Estats diferents: {n_diff} / {total} ({100*n_diff/total:.1f}%)")
    print(f"  Estats iguals:    {total - n_diff} / {total} ({100*(total-n_diff)/total:.1f}%)")
    
    # Mostrar distribució d'accions
    print(f"\nDistribució d'accions:")
    print(f"  VI:  {np.bincount(policy_vi.flatten())}")
    print(f"  Lex: {np.bincount(policy_lex.flatten())}")
    
    # Mostrar exemples d'estats on difereixen
    print(f"\nExemples d'estats on les policies difereixen:")
    diff_indices = np.where(diff)
    for i in range(min(10, len(diff_indices[0]))):
        c = diff_indices[0][i]
        p1 = diff_indices[1][i]
        p2 = diff_indices[2][i]
        print(f"  Estat ({c:2d}, {p1:2d}, {p2:2d}): VI={policy_vi[c,p1,p2]}, Lex={policy_lex[c,p1,p2]}")

# Test amb l'entorn
print("\n" + "="*80)
print("TESTEJANT AMBDUES POLICIES")
print("="*80)

env = Environment(weights=None)

if policy_vi is not None:
    print("\n1. VALUE ITERATION amb pesos [1, 100, 100]:")
    print("-" * 80)
    
    total_r_car = 0
    total_r_ped1 = 0
    total_r_ped2 = 0
    
    for episode in range(10):
        env.reset()
        state = env.get_state()
        done = False
        
        ep_r_car = ep_r_ped1 = ep_r_ped2 = 0
        
        while not done:
            c, p1, p2 = state[0], state[1], state[2]
            action = policy_vi[c, p1, p2]
            next_state, rewards, dones = env.step([action])
            done = dones[0]
            
            ep_r_car += rewards[0]
            ep_r_ped1 += rewards[1]
            ep_r_ped2 += rewards[2]
            
            state = next_state
        
        total_r_car += ep_r_car
        total_r_ped1 += ep_r_ped1
        total_r_ped2 += ep_r_ped2
    
    print(f"  Mean rewards: r_car={total_r_car/10:.2f}, r_ped1={total_r_ped1/10:.2f}, r_ped2={total_r_ped2/10:.2f}")

if policy_lex is not None:
    print("\n2. LEXICOGRAPHIC VI (car priority):")
    print("-" * 80)
    
    total_r_car = 0
    total_r_ped1 = 0
    total_r_ped2 = 0
    
    for episode in range(10):
        env.reset()
        state = env.get_state()
        done = False
        
        ep_r_car = ep_r_ped1 = ep_r_ped2 = 0
        
        while not done:
            c, p1, p2 = state[0], state[1], state[2]
            action = policy_lex[c, p1, p2]
            next_state, rewards, dones = env.step([action])
            done = dones[0]
            
            ep_r_car += rewards[0]
            ep_r_ped1 += rewards[1]
            ep_r_ped2 += rewards[2]
            
            state = next_state
        
        total_r_car += ep_r_car
        total_r_ped1 += ep_r_ped1
        total_r_ped2 += ep_r_ped2
    
    print(f"  Mean rewards: r_car={total_r_car/10:.2f}, r_ped1={total_r_ped1/10:.2f}, r_ped2={total_r_ped2/10:.2f}")

print("\n" + "="*80)