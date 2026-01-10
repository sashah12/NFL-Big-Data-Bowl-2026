import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import torch.nn.functional as F

class PlaySequenceDataset(Dataset):
    def __init__(self, pq_path, feature_cols, off_pos_cols, def_pos_cols):
        self.df = pd.read_parquet(pq_path, memory_map=True, engine='pyarrow')
        self.feature_cols = feature_cols
        self.off_pos_cols = off_pos_cols
        self.def_pos_cols = def_pos_cols
        self.play_groups = self.df.groupby(['game_id', 'play_id']).indices
        self.play_keys = list(self.play_groups.keys())

    def __len__(self):
        return len(self.play_keys)

    def __getitem__(self, idx):
        indices = self.play_groups[self.play_keys[idx]]
        play_df = self.df.iloc[indices].sort_values("frame_id")
        
        X = torch.tensor(play_df[self.feature_cols].values, dtype=torch.float32)
        
        # Extract the 5 position IDs from the first frame (Static for the play)
        # Ensure these are integers for the Embedding layer

        off_ids = torch.tensor(play_df[self.off_pos_cols].iloc[0].values, dtype=torch.long)
        def_ids = torch.tensor(play_df[self.def_pos_cols].iloc[0].values, dtype=torch.long)


        y = torch.tensor([play_df["ball_land_x"].iloc[0], 
                          play_df["ball_land_y"].iloc[0]], dtype=torch.float32)

        route_int = play_df["route_of_targeted_receiver_int"].iloc[0]
        deep_routes = [0, 1, 4, 9] 
        weight = 2.0 if route_int in deep_routes else 1.0
        # --- Inside PlaySequenceDataset.__getitem__ ---
        # Define your weighting strategy (2025 Standard)
        result_map = {
            'C': 1.2,   # Focus more on high-precision successful trajectories
            'I': 0.8,   
            'INT': 1.1

        }
        pass_result = play_df["pass_result"].iloc[0]
        # Use .get() to provide a default weight of 1.0 for unknown results
        result_weight = result_map.get(pass_result, 1.0)
        
        # Combine this with your existing route-based weight (e.g., 1.5 for deep)
        # Total weight = Route Difficulty * Play Result Importance
        final_weight = weight * result_weight 
        return X, off_ids, def_ids, y, torch.tensor(weight, dtype=torch.float32)

def collate_fn(batch):
    X_list, off_list, def_list, y_list, w_list = zip(*batch)
    X_padded = pad_sequence(X_list, batch_first=True, padding_value=0.0)
    
    mask = torch.zeros(X_padded.size(0), X_padded.size(1), dtype=torch.bool)
    for i, x in enumerate(X_list):
        mask[i, :x.size(0)] = True 

    return X_padded, torch.stack(off_list), torch.stack(def_list), mask, torch.stack(y_list), torch.stack(w_list)


