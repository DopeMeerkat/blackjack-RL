import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
import numpy as np

def cmp(a, b):
    return float(a > b) - float(a < b)

# The deck is infinite (drawn with replacement). 
# 1 = Ace, 2-10 = Number cards, 10 = Face cards.
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

def draw_card(np_random):
    return int(np_random.choice(deck))

def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]

def usable_ace(hand):  
    # Does this hand have an Ace that can be counted as 11 without busting?
    return 1 in hand and sum(hand) + 10 <= 21

def sum_hand(hand): 
    # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)

def is_bust(hand):
    return sum_hand(hand) > 21

def score(hand): 
    # What is the final score? (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)

class BlackjackEnvV2(gym.Env):
    """
    Custom Blackjack Environment supporting Hit, Stand, Split, and Double Down.
    
    Action Space:
        0: Hit
        1: Stand
        2: Split
        3: Double Down
        
    Observation Space: Tuple of 5 Discrete values
        - Player's current sum (0-31)
        - Dealer's showing card (1-10)
        - Usable Ace (0 or 1)
        - Can Split (0 or 1)
        - Can Double Down (0 or 1)
    """
    
    metadata = {
        "render_modes": ["human", "ansi"],
        "render_fps": 2,
    }

    def __init__(self, render_mode=None):
        self.action_space = spaces.Discrete(4)
        
        # obs: (player_sum, dealer_card, usable_ace, can_split, can_double)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2),
            spaces.Discrete(2),
            spaces.Discrete(2)
        ))
        
        self.render_mode = render_mode
        self.dealer = []
        self.hands = []
        self.bets = []
        self.current_hand_index = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.dealer = draw_hand(self.np_random)
        self.hands = [draw_hand(self.np_random)]
        self.bets = [1.0]
        self.current_hand_index = 0
        
        return self._get_obs(), self._get_info()

    def _get_info(self):
        """Returns a dictionary containing the action mask for the current state."""
        if self.current_hand_index >= len(self.hands):
            return {"action_mask": np.array([0, 0, 0, 0], dtype=np.int8)}

        active_hand = self.hands[self.current_hand_index]
        can_split = int(len(active_hand) == 2 and (active_hand[0] == active_hand[1]))
        can_double = int(len(active_hand) == 2)
        # 1 means valid, 0 means invalid. Hit(0) and Stand(1) are always valid.
        return {"action_mask": np.array([1, 1, can_split, can_double], dtype=np.int8)}

    def _get_obs(self):
        active_hand = self.hands[self.current_hand_index]
        can_split = int(len(active_hand) == 2 and active_hand[0] == active_hand[1])
        can_double = int(len(active_hand) == 2)
        
        return (
            sum_hand(active_hand),
            self.dealer[0],
            int(usable_ace(active_hand)),
            can_split,
            can_double
        )

    def step(self, action):
        assert self.action_space.contains(action)
        
        active_hand = self.hands[self.current_hand_index]
        info = self._get_info()
        mask = info["action_mask"]
        
        # STRICT RULE ENFORCEMENT
        if mask[action] == 0:
            # Agent chose an illegal action. End game immediately with maximum penalty.
            return self._get_obs(), -1.0, True, False, info
            
        hand_done = False
        
        if action == 0:  # Hit
            active_hand.append(draw_card(self.np_random))
            if is_bust(active_hand):
                hand_done = True
                
        elif action == 1:  # Stand
            hand_done = True
            
        elif action == 2:  # Split
            split_card = active_hand.pop()
            new_hand = [split_card, draw_card(self.np_random)]
            self.hands.insert(self.current_hand_index + 1, new_hand)
            self.bets.insert(self.current_hand_index + 1, 1.0)
            active_hand.append(draw_card(self.np_random))
                
        elif action == 3:  # Double Down
            self.bets[self.current_hand_index] *= 2.0
            active_hand.append(draw_card(self.np_random))
            hand_done = True

        if hand_done:
            self.current_hand_index += 1

        terminated = self.current_hand_index >= len(self.hands)
        reward = 0.0

        if terminated:
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            
            dealer_score = score(self.dealer)
            for hand, bet in zip(self.hands, self.bets):
                player_score = score(hand)
                if player_score == 0:
                    reward -= bet
                elif dealer_score == 0:
                    reward += bet
                else:
                    reward += cmp(player_score, dealer_score) * bet

        if self.render_mode == "human":
            self.render()

        return self._get_obs() if not terminated else (0, 0, 0, 0, 0), reward, terminated, False, self._get_info()

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("You are calling render method without specifying any render mode.")
            return

        outfile = ""
        
        # Render Dealer
        outfile += f"Dealer: "
        if self.current_hand_index >= len(self.hands):
            # Game over, show all dealer cards
            outfile += f"{self.dealer} (Sum: {sum_hand(self.dealer)})\n"
        else:
            # Game active, hide second card
            outfile += f"[{self.dealer[0]}, ?]\n"
        
        # Render Player Hands
        outfile += "Player Hands:\n"
        for i, hand in enumerate(self.hands):
            active_marker = " <--" if i == self.current_hand_index else ""
            status = " (Bust)" if is_bust(hand) else f" (Sum: {sum_hand(hand)})"
            bet_status = f" [Bet: {self.bets[i]}]"
            outfile += f"  Hand {i + 1}: {hand}{status}{bet_status}{active_marker}\n"
            
        outfile += "-" * 30 + "\n"

        if self.render_mode == "human":
            print(outfile)
        return outfile