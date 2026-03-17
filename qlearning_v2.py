import numpy as np
import random
from collections import defaultdict
from blackjack_v2 import BlackjackEnvV2

def train_blackjack_agent():
    # 1. Initialize the environment
    env = BlackjackEnvV2()
    
    # 2. Initialize the Q-table
    # A defaultdict allows us to start with an array of zeros [0,0,0,0] 
    # for any new state the agent encounters.
    q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # 3. Hyperparameters
    num_episodes = 500_000      # Number of games to play
    alpha = 0.01                # Learning rate
    gamma = 0.95                # Discount factor (future reward importance)
    epsilon = 1.0               # Initial exploration rate
    epsilon_min = 0.05          # Minimum exploration rate
    epsilon_decay = 0.99999     # Rate at which epsilon decays
    
    print("Starting training...")
    
    # 4. Training Loop
    for episode in range(num_episodes):
        state, info = env.reset() # Get initial info
        done = False
        
        while not done:
            mask = info["action_mask"]
            
            # Epsilon-greedy action selection WITH MASKING
            if random.uniform(0, 1) < epsilon:
                # Explore: choose randomly ONLY from valid actions
                valid_actions = [i for i, valid in enumerate(mask) if valid == 1]
                action = random.choice(valid_actions)
            else:
                # Exploit: apply mask to Q-values to ignore invalid actions
                q_values = np.copy(q_table[state])
                q_values[mask == 0] = -np.inf # Set invalid actions to negative infinity
                action = np.argmax(q_values)
                
            next_state, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            
            # Q-Learning update rule WITH MASKING for the next state
            next_mask = next_info["action_mask"]
            next_q_values = np.copy(q_table[next_state])
            next_q_values[next_mask == 0] = -np.inf 
            
            best_next_action = np.argmax(next_q_values) if not done else 0
            
            td_target = reward + gamma * q_table[next_state][best_next_action] * (not done)
            td_error = td_target - q_table[state][action]
            
            q_table[state][action] += alpha * td_error
            
            state = next_state
            info = next_info
            
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Optional: Print progress
        if (episode + 1) % 100_000 == 0:
            print(f"Episode {episode + 1}/{num_episodes} completed. Epsilon: {epsilon:.3f}")

    print("Training finished!\n")
    return q_table

def evaluate_agent(q_table, num_games=5):
    """Watch the trained agent play a few games."""
    # Turn on rendering to watch it play
    env = BlackjackEnvV2(render_mode="human")
    
    action_names = {0: "Hit", 1: "Stand", 2: "Split", 3: "Double Down"}
    
    for i in range(num_games):
        print(f"=== GAME {i+1} ===")
        state, info = env.reset()
        done = False
        
        env.render()
        while not done:
            mask = info["action_mask"]

            # Exploit: apply mask to Q-values to ignore invalid actions
            q_values = np.copy(q_table[state])
            q_values[mask == 0] = -np.inf # Set invalid actions to negative infinity
            action = np.argmax(q_values)
            print(f"Agent chooses: **{action_names[action]}**\n")
            
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        print(f"Game over. Reward: {reward}\n")

def evaluate_win_rate(q_table, num_games=10_000):
    """Calculate the expected value and win rate of the trained agent."""
    # Use a non-rendering environment for fast simulation
    env = BlackjackEnvV2()
    
    print(f"\nEvaluating agent over {num_games} games...")
    total_reward = 0.0
    wins = 0
    losses = 0
    draws = 0

    for _ in range(num_games):
        state, info = env.reset()
        done = False
        
        while not done:
            mask = info["action_mask"]
            # Exploit: apply mask to Q-values to ignore invalid actions
            q_values = np.copy(q_table[state])
            q_values[mask == 0] = -np.inf # Set invalid actions to negative infinity
            action = np.argmax(q_values)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        total_reward += reward
        
        # Categorize the result
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    # Calculate statistics
    ev = total_reward / num_games
    win_rate = (wins / num_games) * 100
    loss_rate = (losses / num_games) * 100
    draw_rate = (draws / num_games) * 100

    print("-" * 30)
    print(f"Results after {num_games} games:")
    print(f"Win Rate:  {win_rate:.2f}%")
    print(f"Loss Rate: {loss_rate:.2f}%")
    print(f"Draw Rate: {draw_rate:.2f}%")
    print(f"Expected Value (EV) per hand: {ev:.4f}")
    print("-" * 30)
    
    return ev

def print_strategy_chart(q_table):
    """Prints a visual Basic Strategy chart for Hard Totals based on the learned Q-Table."""
    print("\n--- Learned Basic Strategy: Hard Totals ---")
    print("H = Hit, S = Stand, D = Double Down")
    print("-------------------------------------------")
    print("Player | Dealer Showing Card")
    print(" Sum   | 2  3  4  5  6  7  8  9  10 A")
    print("-------+-----------------------------------")
    
    action_symbols = {0: 'H', 1: 'S', 2: 'P', 3: 'D'}
    
    # Check player totals from 21 down to 8
    for player_sum in range(17, 7, -1):
        row = f"  {player_sum:2d}   | "
        
        # Check dealer cards 2-10, and Ace (represented as 1 in our env)
        for dealer_card in [2, 3, 4, 5, 6, 7, 8, 9, 10, 1]:
            # State: (player_sum, dealer_card, usable_ace=0, can_split=0, can_double=1)
            state = (player_sum, dealer_card, 0, 0, 1)
            
            # Apply mask: Cannot split on hard totals
            q_values = np.copy(q_table[state])
            q_values[2] = -np.inf 
            
            best_action = np.argmax(q_values)
            
            # Highlight Doubles with brackets for readability
            symbol = action_symbols[best_action]
            if symbol == 'D':
                row += "[D]"
            else:
                row += f" {symbol} "
                
        print(row)

def print_soft_totals_chart(q_table):
    """Prints a visual Basic Strategy chart for Soft Totals based on the learned Q-Table."""
    print("\n--- Learned Basic Strategy: Soft Totals (Usable Ace) ---")
    print("H = Hit, S = Stand, D = Double Down")
    print("--------------------------------------------------------")
    print("Player | Dealer Showing Card")
    print(" Hand  | 2  3  4  5  6  7  8  9  10 A")
    print("-------+-----------------------------------")
    
    action_symbols = {0: 'H', 1: 'S', 2: 'P', 3: 'D'}
    
    # Soft totals typically range from 21 (A, 10) down to 13 (A, 2)
    for player_sum in range(21, 12, -1):
        # Format the row label (e.g., A,7 for 18)
        if player_sum == 21:
            row_label = " A,10 "
        else:
            row_label = f" A,{player_sum - 11:<2}"
            
        row = f" {row_label} | "
        
        # Check dealer cards 2-10, and Ace (represented as 1)
        for dealer_card in [2, 3, 4, 5, 6, 7, 8, 9, 10, 1]:
            # 1. Query the state where doubling IS allowed
            state_double_allowed = (player_sum, dealer_card, 1, 0, 1)
            q_vals_allowed = np.copy(q_table[state_double_allowed])
            q_vals_allowed[2] = -np.inf # Cannot split
            
            best_action = np.argmax(q_vals_allowed)
            symbol = ""
            
            if best_action == 3:  # Double Down
                # 2. Query the fallback state where doubling is NOT allowed (e.g. 3+ cards)
                state_no_double = (player_sum, dealer_card, 1, 0, 0)
                q_vals_not_allowed = np.copy(q_table[state_no_double])
                q_vals_not_allowed[2] = -np.inf # Cannot split
                q_vals_not_allowed[3] = -np.inf # Cannot double down
                
                fallback_action = np.argmax(q_vals_not_allowed)
                
                if fallback_action == 0:  # Fallback is Hit
                    symbol = "D"
                else:                     # Fallback is Stand
                    symbol = "Ds"
            elif best_action == 0:
                symbol = "H"
            else:
                symbol = "S"
                
            # Format to take up 4 spaces for clean column alignment
            row += f" {symbol:2} "
                
        print(row)

def print_pairs_chart(q_table):
    """Prints a visual Basic Strategy chart for Pairs based on the learned Q-Table."""
    print("\n--- Learned Basic Strategy: Pairs ---")
    print("H = Hit, S = Stand, P = Split, D = Double Down")
    print("----------------------------------------------")
    print("Player | Dealer Showing Card")
    print(" Pair  | 2  3  4  5  6  7  8  9  10 A")
    print("-------+-----------------------------------")
    
    action_symbols = {0: 'N', 1: 'N', 2: 'Y', 3: 'N'}
    
    # Define the pairs, their sums, and if they contain a usable ace
    pairs = [
        ("A,A", 12, 1), ("10,10", 20, 0), ("9,9", 18, 0), 
        ("8,8", 16, 0), ("7,7", 14, 0), ("6,6", 12, 0), 
        ("5,5", 10, 0), ("4,4", 8, 0), ("3,3", 6, 0), ("2,2", 4, 0)
    ]
    
    for label, player_sum, usable_ace in pairs:
        row = f" {label:5} | "
        
        # Check dealer cards 2-10, and Ace (represented as 1)
        for dealer_card in [2, 3, 4, 5, 6, 7, 8, 9, 10, 1]:
            # State: (player_sum, dealer_card, usable_ace, can_split=1, can_double=1)
            state = (player_sum, dealer_card, usable_ace, 1, 1)
            
            # All 4 actions are valid on a fresh pair, so no mask is needed to filter the Q-values
            q_values = q_table[state]
            best_action = np.argmax(q_values)
            
            # Highlight Splits and Doubles with brackets
            symbol = action_symbols[best_action]
            if symbol == 'Y':
                row += "[Y]"
            else:
                row += f" {symbol} "
                
        print(row)

if __name__ == "__main__":
    # 1. Train the Q-table
    trained_q_table = train_blackjack_agent()
    
    # 2. Watch the agent play a few visual games
    evaluate_agent(trained_q_table, num_games=3)
    
    # 3. Calculate statistically significant win/loss metrics
    evaluate_win_rate(trained_q_table, num_games=10_000)

    # 4. Print the learned Q-values for inspection
    print_strategy_chart(trained_q_table)
    print_soft_totals_chart(trained_q_table)
    print_pairs_chart(trained_q_table)