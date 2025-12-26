from typing import List, Dict, Any
import math

class ShotSelector:
    def __init__(self, strategy: str = "greedy"):
        self.strategy = strategy

    def select(self, shots: List[Dict], scores: List[float], target_duration: float) -> List[Dict]:
        """
        Select shots based on scores and target duration.
        shots: List of dicts with 'start_sec', 'end_sec', 'duration_sec'.
        scores: List of importance scores (0-1).
        """
        # Attach scores to shots
        scored_shots = []
        for shot, score in zip(shots, scores):
            s = shot.copy()
            s['score'] = score
            scored_shots.append(s)
            
        if self.strategy == "greedy":
            return self._greedy_selection(scored_shots, target_duration)
        elif self.strategy == "knapsack":
            return self._knapsack_selection(scored_shots, target_duration)
        else:
            # Default greedy
            return self._greedy_selection(scored_shots, target_duration)

    def _greedy_selection(self, shots: List[Dict], limit: float) -> List[Dict]:
        # Sort by score descending
        sorted_shots = sorted(shots, key=lambda x: x['score'], reverse=True)
        
        selected = []
        current_duration = 0.0
        
        for shot in sorted_shots:
            if current_duration + shot['duration_sec'] <= limit:
                selected.append(shot)
                current_duration += shot['duration_sec']
                
        # Sort back by time
        selected.sort(key=lambda x: x['start_sec'])
        return selected

    def _knapsack_selection(self, shots: List[Dict], limit: float) -> List[Dict]:
        # 0/1 Knapsack
        # Value = score * duration (to maximize total importance-time) or just score?
        # Spec says: "Treat each shot as item with value=score and weight=duration"
        # Since duration is float, we need to discretize or use DP with scaling.
        # Let's scale duration to seconds (int) or deciseconds.
        
        scale = 10 # 0.1s precision
        capacity = int(limit * scale)
        n = len(shots)
        weights = [int(s['duration_sec'] * scale) for s in shots]
        values = [s['score'] for s in shots] # Just score
        
        # DP table: dp[i][w] = max value using first i items with weight w
        # This can be huge if capacity is large.
        # If limit is 300s -> 3000 units. Manageable.
        
        dp = [[0.0] * (capacity + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            w = weights[i-1]
            v = values[i-1]
            for j in range(capacity + 1):
                if w <= j:
                    dp[i][j] = max(dp[i-1][j], dp[i-1][j-w] + v)
                else:
                    dp[i-1][j]
                    
        # Backtrack to find items
        selected_indices = []
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                selected_indices.append(i-1)
                w -= weights[i-1]
                
        selected = [shots[i] for i in selected_indices]
        selected.sort(key=lambda x: x['start_sec'])
        return selected
