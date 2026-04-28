import csv
import numpy as np
from blackjack_v2 import BlackjackEnvV2

# Actions in BlackjackEnvV2:
# 0: Hit, 1: Stand, 2: Split, 3: Double Down

DEALER_COLS = ["2","3","4","5","6","7","8","9","10","A"]

def load_strategy_csv(path: str, key_col: str):
    """
    Loads a strategy table CSV into:
      table[key][dealer_upcard] = action_symbol (e.g., 'H','S','D','Ds','Y','N','Y/N')
    """
    table = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if key_col not in reader.fieldnames:
            raise ValueError(f"{path}: expected key_col='{key_col}' in header, got {reader.fieldnames}")
        for row in reader:
            key = row[key_col].strip()
            table[key] = {d: row[d].strip() for d in DEALER_COLS}
    return table

def dealer_key(dealer_card: int) -> str:
    # In this env, Ace is 1
    return "A" if dealer_card == 1 else str(dealer_card)

def infer_pair_label(player_sum: int, usable_ace: int) -> str | None:
    """
    For a splittable 2-card pair, infer the pair label (AA, TT, 99, ..., 22)
    from (player_sum, usable_ace). This works because pairs map uniquely:
      AA -> (12, usable_ace=1)
      22 -> (4,0), 33 -> (6,0), ..., TT -> (20,0)
    """
    if usable_ace == 1 and player_sum == 12:
        return "AA"
    mapping = {
        4: "22",
        6: "33",
        8: "44",
        10: "55",
        12: "66",
        14: "77",
        16: "88",
        18: "99",
        20: "TT",
    }
    return mapping.get(player_sum)

def symbol_to_action(symbol: str, can_double: int) -> int:
    """
    Converts strategy symbols to env actions.
      H  -> Hit
      S  -> Stand
      D  -> Double if allowed else Hit
      Ds -> Double if allowed else Stand
    """
    symbol = symbol.strip()
    if symbol == "H":
        return 0
    if symbol == "S":
        return 1
    if symbol == "D":
        return 3 if can_double else 0
    if symbol == "Ds":
        return 3 if can_double else 1
    raise ValueError(f"Unknown symbol: {symbol}")

def choose_action(state, info, tables, das: bool = True) -> int:
    """
    Applies S17 basic strategy (with DAS support) to BlackjackEnvV2 states:
      state = (player_sum, dealer_upcard, usable_ace, can_split, can_double)
    """
    player_sum, dealer_card, usable_ace, can_split, can_double = state
    dkey = dealer_key(dealer_card)

    # Ensure mask is a numpy array
    mask = info.get("action_mask", np.array([1, 1, 0, 0], dtype=np.int8))
    mask = np.asarray(mask, dtype=np.int8)

    pairs_table, soft_table, hard_table = tables["pairs"], tables["soft"], tables["hard"]

    # 1) Pair splitting decisions (only when split is legal per env)
    if int(can_split) == 1:
        pair_label = infer_pair_label(int(player_sum), int(usable_ace))
        if pair_label is not None and pair_label in pairs_table:
            decision = pairs_table[pair_label][dkey]
            if decision == "Y/N":
                decision = "Y" if das else "N"
            if decision == "Y" and mask[2] == 1:
                return 2  # Split
            # If decision is N, fall through to totals strategy (e.g., 5,5 -> treat as hard 10)

    # 2) Soft vs Hard totals
    # Soft table in the chart covers totals 13-21 when usable_ace==1
    if int(usable_ace) == 1 and str(int(player_sum)) in soft_table:
        sym = soft_table[str(int(player_sum))][dkey]
        a = symbol_to_action(sym, int(can_double) == 1 and mask[3] == 1)
    else:
        # Default hard total rules if not explicitly in table:
        # <= 8 -> Hit, >= 17 -> Stand
        ps = int(player_sum)
        if ps <= 8:
            sym = "H"
        elif ps >= 17:
            sym = "S"
        else:
            sym = hard_table[str(ps)][dkey] if str(ps) in hard_table else "H"
        a = symbol_to_action(sym, int(can_double) == 1 and mask[3] == 1)

    # Final legality guard (should rarely trigger if tables + env match)
    if a < 0 or a >= 4 or mask[a] == 0:
        # Prefer Stand if legal, else Hit, else any legal action.
        for fallback in [1, 0, 3, 2]:
            if fallback < len(mask) and mask[fallback] == 1:
                return fallback
        return 0

    return a

def evaluate_agent(tables, num_games=3, das: bool = True):
    """Watch the basic-strategy agent play a few games."""
    env = BlackjackEnvV2(render_mode="human")
    action_names = {0: "Hit", 1: "Stand", 2: "Split", 3: "Double Down"}

    for i in range(num_games):
        print(f"=== GAME {i+1} ===")
        state, info = env.reset()
        done = False

        env.render()
        while not done:
            action = choose_action(state, info, tables, das=das)
            print(f"Agent chooses: {action_names[action]}")
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        print(f"Game over. Reward: {reward}\n")

def evaluate_win_rate(tables, num_games=100_000, das: bool = True, seed: int = 0):
    """Compute EV and win/loss/draw rates under basic strategy."""
    env = BlackjackEnvV2()
    rng = np.random.default_rng(seed)

    total_reward = 0.0
    wins = losses = draws = 0

    for _ in range(num_games):
        state, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        reward = 0.0

        while not done:
            action = choose_action(state, info, tables, das=das)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        total_reward += float(reward)
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    ev = total_reward / num_games
    print("-" * 34)
    print(f"Basic Strategy (S17, DAS={das}) over {num_games} games")
    print(f"Win Rate:  {wins/num_games*100:.2f}%")
    print(f"Loss Rate: {losses/num_games*100:.2f}%")
    print(f"Draw Rate: {draws/num_games*100:.2f}%")
    print(f"EV/hand:   {ev:.5f}")
    print("-" * 34)
    return ev

def main():
    tables = {
        "pairs": load_strategy_csv("pairs_s17_das.csv", key_col="pair"),
        "soft":  load_strategy_csv("soft_totals_s17.csv", key_col="total"),
        "hard":  load_strategy_csv("hard_totals_s17.csv", key_col="total"),
    }

    # Watch a few games (requires a GUI-capable session)
    evaluate_agent(tables, num_games=3, das=True)

    # Then evaluate statistically (headless)
    evaluate_win_rate(tables, num_games=50_000, das=True, seed=5820)

if __name__ == "__main__":
    main()
