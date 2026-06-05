from src.minesweeper import MineSweeper
import pygame
import time
from pyDatalog import pyDatalog

# ------------------------------------------------------------------
# 1. Define Terms
# ------------------------------------------------------------------
# Vocabulary: R/C (Row/Col), V (Value), FC/HC (Flag/Hidden Counts)
pyDatalog.create_terms('R, C, NR, NC, V, FC, HC')
pyDatalog.create_terms('revealed, hidden, flagged, neighbor, clue')
pyDatalog.create_terms('flagged_count, hidden_count')
pyDatalog.create_terms('safe, mine, safe_zero, frontier')  # added frontier

def prolog_solver(game):
    try:
        print("Initializing Static Facts...")

        # Init empty predicates to prevent engine errors
        + clue(0,0,0); - clue(0,0,0)
        + flagged_count(0,0,0); - flagged_count(0,0,0)
        + flagged(0,0); - flagged(0,0)
        + revealed(0,0); - revealed(0,0)

        # ------------------------------------------------------------------
        # 3. Define Facts
        # ------------------------------------------------------------------
        # Static setup: Map board topology (neighbors) and set initial hidden state
        for r in range(game.rows):
            for c in range(game.cols):
                + hidden(r, c)
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < game.rows and 0 <= nc < game.cols:
                            + neighbor(r, c, nr, nc)

        # ------------------------------------------------------------------
        # 2. Define Rules
        # ------------------------------------------------------------------
        # Logic Rule 1 (Safe): If Flagged neighbors == Clue -> Remaining hidden are Safe
        safe(NR, NC) <= (clue(R, C, V) & (V > 0) & flagged_count(R, C, FC) & (FC == V) &  # CHANGED: added (V > 0)
                         hidden_count(R, C, HC) & (HC > 0) & neighbor(R, C, NR, NC) & hidden(NR, NC))
        
        # Logic Rule 2 (Mine): If Flagged + Hidden neighbors == Clue -> Remaining hidden are Mines
        mine(NR, NC) <= (clue(R, C, V) & (V > 0) & flagged_count(R, C, FC) & hidden_count(R, C, HC) &  # CHANGED: added (V > 0)
                         (FC + HC == V) & (HC > 0) & neighbor(R, C, NR, NC) & hidden(NR, NC))
        
        # Logic Rule 3 (Safe Zero): If clue == 0 -> All neighbors are safe
        safe_zero(NR, NC) <= (clue(R, C, 0) & neighbor(R, C, NR, NC) & hidden(NR, NC))
        
        # Logic Rule 4: Frontier Rule - Only revealed cells with at least one hidden neighbor
        frontier(R, C) <= (revealed(R, C) & neighbor(R, C, NR, NC) & hidden(NR, NC))

        # Start game
        start_r, start_c = game.get_start_pos()
        clue_value = game.reveal(start_r, start_c)
        
        if clue_value is None:
            clue_value = 0
        
        if clue_value != -1:
            - hidden(start_r, start_c)
            + revealed(start_r, start_c)
            + clue(start_r, start_c, clue_value)
        
        running = True
        while running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False

            game.render()
            if game.game_over:
                time.sleep(2)
                break

            # ------------------------------------------------------------------
            # 6. Assert New Facts
            # ------------------------------------------------------------------
            # Helper: Aggregate neighbor counts (Flags/Hidden) for current cycle
            current_cycle_facts = []

            # Use frontier instead of revealed for efficiency
            frontier_cells = frontier(R, C)
            
            if frontier_cells:
                for cell in list(frontier_cells):
                    r, c = cell[0], cell[1]
                    f_count = 0
                    h_count = 0
                    
                    # count neighbors manually for speed
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < game.rows and 0 <= nc < game.cols:
                                is_flagged = len(list(flagged(nr, nc))) > 0
                                is_hidden = len(list(hidden(nr, nc))) > 0
                                
                                if is_flagged:
                                    f_count += 1
                                elif is_hidden:
                                    h_count += 1
                    
                    # Assert temporary counts
                    + flagged_count(r, c, f_count)
                    + hidden_count(r, c, h_count)
                    current_cycle_facts.append((r, c, f_count, h_count))

            # ------------------------------------------------------------------
            # 4. Query to Reveal/Flag
            # ------------------------------------------------------------------
            # Batch actions instead of immediate execution
            actions_batch = []
            
            # Priority 1 - Query safe_zero cells first
            safe_zero_cells = safe_zero(R, C)
            if safe_zero_cells:
                for cell in list(safe_zero_cells):
                    r, c = cell[0], cell[1]
                    is_hidden = len(list(hidden(r, c))) > 0
                    if is_hidden:
                        if len(actions_batch) >= 20:  # early termination
                            break
                        actions_batch.append(('reveal', r, c))
            
            # Priority 2 - Query safe cells
            if len(actions_batch) < 20:  # check batch limit
                safe_cells = safe(R, C)
                if safe_cells:
                    for cell in list(safe_cells):
                        r, c = cell[0], cell[1]
                        is_hidden = len(list(hidden(r, c))) > 0
                        if is_hidden:
                            if len(actions_batch) >= 20:  # early termination
                                break
                            actions_batch.append(('reveal', r, c))
            
            # ------------------------------------------------------------------
            # 5. Take Actions
            # ------------------------------------------------------------------
            # Priority 3 - Query mine cells (only if no safe moves)
            if len(actions_batch) == 0:  # check if batch is empty
                mine_cells = mine(R, C)
                if mine_cells:
                    for cell in list(mine_cells):
                        r, c = cell[0], cell[1]
                        is_hidden = len(list(hidden(r, c))) > 0
                        if is_hidden:
                            if len(actions_batch) >= 20:  # early termination
                                break
                            actions_batch.append(('flag', r, c))
            
            # Priority 4 - Smart guessing when no logical moves available
            if len(actions_batch) == 0:
                # Get all hidden cells using PyDatalog query
                all_hidden = hidden(R, C)
                
                if all_hidden:
                    hidden_list = list(all_hidden)
                    best_cell = None
                    lowest_probability = float('inf')
                    
                    # Calculate mine probability for each hidden cell (limit to first 50 for performance)
                    for cell in hidden_list[:50]:
                        r, c = cell[0], cell[1]
                        
                        # Find revealed neighbors using PyDatalog
                        total_probability = 0
                        neighbor_count = 0
                        
                        # Get neighbors from fact
                        cell_neighbors = neighbor(r, c, NR, NC)
                        if cell_neighbors:
                            for neighbor_cell in list(cell_neighbors):
                                nr, nc = neighbor_cell[0], neighbor_cell[1]
                                
                                # Check if this neighbor is revealed
                                is_revealed = len(list(revealed(nr, nc))) > 0
                                
                                if is_revealed:
                                    # Get clue value from fact
                                    clue_result = clue(nr, nc, V)
                                    if clue_result:
                                        clue_value = list(clue_result)[0][0]
                                        
                                        # Count flagged and hidden neighbors of this revealed cell
                                        flagged_count_val = 0
                                        hidden_count_val = 0
                                        
                                        neighbor_neighbors = neighbor(nr, nc, NR, NC)
                                        if neighbor_neighbors:
                                            for nn_cell in list(neighbor_neighbors):
                                                nnr, nnc = nn_cell[0], nn_cell[1]
                                                
                                                if len(list(flagged(nnr, nnc))) > 0:
                                                    flagged_count_val += 1
                                                elif len(list(hidden(nnr, nnc))) > 0:
                                                    hidden_count_val += 1
                                        
                                        # Calculate local probability
                                        if hidden_count_val > 0:
                                            local_prob = (clue_value - flagged_count_val) / hidden_count_val
                                            total_probability += local_prob
                                            neighbor_count += 1
                        
                        # Calculate average probability
                        if neighbor_count > 0:
                            avg_probability = total_probability / neighbor_count
                            if avg_probability < lowest_probability:
                                lowest_probability = avg_probability
                                best_cell = (r, c)
                    
                    # If we found a cell with calculated probability, use it
                    if best_cell:
                        actions_batch.append(('reveal', best_cell[0], best_cell[1]))
                        print(f"\n[Guess: prob={lowest_probability:.2f}]", end="", flush=True)
                    # Otherwise, just pick the first hidden cell
                    elif hidden_list:
                        r, c = hidden_list[0][0], hidden_list[0][1]
                        actions_batch.append(('reveal', r, c))
                        print("\n[Random guess]", end="", flush=True)
            
            # Execute batch actions
            found_action = False
            for action_type, r, c in actions_batch:
                if action_type == 'reveal':
                    clue_value = game.reveal(r, c)
                    
                    if clue_value is None:
                        continue
                    
                    if clue_value == -1:
                        print(f"\n[MINE at ({r},{c})]")
                        break
                    
                    - hidden(r, c)
                    + revealed(r, c)
                    + clue(r, c, clue_value)
                    found_action = True
                    
                elif action_type == 'flag':
                    game.flag(r, c)
                    - hidden(r, c)
                    + flagged(r, c)
                    found_action = True

            # Cleanup: Retract temporary counts
            for r, c, f, h in current_cycle_facts:
                - flagged_count(r, c, f)
                - hidden_count(r, c, h)

            # UI Feedback
            if not found_action:
                print(".", end="", flush=True)
            else:
                print(f"!{len(actions_batch)}", end="", flush=True)
            
        print("\nProgram completed successfully!")
        
    except Exception as e:
        print(f"\nCRASH DETECTED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # ms = MineSweeper(rows=9, cols=9, mines=9, seed=99, auto_flood_fill=False)
    # ms = MineSweeper(rows=15, cols=15, mines=35, seed=42, auto_flood_fill=False)
    # ms = MineSweeper(rows=20, cols=20, mines=75, seed=123, auto_flood_fill=False)
    ms = MineSweeper(rows=45, cols=80, mines=200, seed=-1, auto_flood_fill=False)
    prolog_solver(ms)