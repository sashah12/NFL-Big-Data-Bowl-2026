import os
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader

from dataset import PlaySequenceDataset, collate_fn  # <-- dataset
from transf import PlayTransformer                    # <-- transformer model
from torch.optim.lr_scheduler import OneCycleLR

def main(FEATURES):

    PQ_PATH = "train_df_scaled.parquet"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 64               
    EPOCHS = 20                  
    LR = 1e-4                     
    ACCUM_STEPS = 2               
    WEIGHT_DECAY = 0.10           
    WARMUP_STEPS = 1000           
    NUM_WORKERS = 0   
    CHECKPOINT_EVERY = 5

    
    # total_steps = (len(train_loader) // ACCUM_STEPS) * EPOCHS

    # ==============================
    # 1. SETUP DATA 
    train_dataset = PlaySequenceDataset(PQ_PATH, FEATURES, ['off1_player_position_idx', 'off2_player_position_idx', 'off3_player_position_idx', 
                                                            'off4_player_position_idx', 'off5_player_position_idx'], ['def1_player_position_idx', 
                                                            'def2_player_position_idx', 'def3_player_position_idx', 'def4_player_position_idx', 
                                                            'def5_player_position_idx', 'def6_player_position_idx', 'def7_player_position_idx', 
                                                            'def8_player_position_idx', 'def9_player_position_idx', 'def10_player_position_idx', 
                                                            'def11_player_position_idx'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS, 
        pin_memory=True,         
        collate_fn=collate_fn
    )

    # 2. SETUP MODEL
    model = PlayTransformer(continuous_dim=len(FEATURES)).to(DEVICE)

    # 3. SETUP OPTIMIZER 
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LR, 
        weight_decay=WEIGHT_DECAY
    )
    
    # 4. SETUP SCHEDULER 
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=3e-4, 
        steps_per_epoch=len(train_loader) // ACCUM_STEPS, 
        epochs=EPOCHS,
        pct_start=0.1 
    )

    # 5. SETUP SCALER
    scaler = torch.amp.GradScaler(device="cuda")

    # ==============================
    # TRAIN LOOP
    # ==============================
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_ball_loss = 0
        total_ball_loss_rmse = 0
        batches = 0
        # 1. Initialize Huber Loss outside the loop
        # delta=1 treats errors > 1 yards linearly (prevents gradient spikes)
        huber_fn = torch.nn.HuberLoss(delta=1.0, reduction='none')
        
        for i, (X, off_ids, def_ids, mask, ball_target, weight) in enumerate(train_loader):
            X, off_ids, def_ids, mask, ball_target, weight = X.to(DEVICE), off_ids.to(DEVICE), def_ids.to(DEVICE), mask.to(DEVICE), ball_target.to(DEVICE), weight.to(DEVICE)
        
            with torch.amp.autocast(device_type="cuda"):
                # --- 1. FORWARD PASS ---
                preds = model(X, off_ids, def_ids, mask=mask)["ball_location"] # Shape: (B, Seq, 2)
                target_expanded = ball_target.unsqueeze(1).expand(-1, X.size(1), -1)
        
                # --- 2. BOUNDARY WEIGHT CALCULATION ---
                # Calculate proximity to sidelines (Y) and endzones (X)
                dist_y = torch.min(ball_target[:, 1], 53.3 - ball_target[:, 1])
                dist_x = torch.min(ball_target[:, 0], 120.0 - ball_target[:, 0])
                
                # Exponential ramp (7.0 decay is ideal for football spatial cushions)
                # Weight is ~2.0 at boundary, ~1.0 in middle of field
                b_weight = torch.exp(-torch.min(dist_x, dist_y) / 3.0) + 1.0
                
                # --- 3. HUBER LOSS CALCULATION ---
                # Compute raw Huber loss per coordinate, then average for (X, Y)
                # Shape: (B, Seq)
                raw_huber = huber_fn(preds, target_expanded).mean(dim=-1)
                
                # Apply boundary weight and mask
                weighted_huber = (raw_huber * mask.float()) * (weight * b_weight).unsqueeze(1)
        
                # --- 4. TWO-HEADED TASK SEPARATION ---
                # Deep routes (weight > 1.0) vs Standard routes (weight <= 1.0)
                deep_mask = (weight > 1.0).unsqueeze(1) * mask.float()
                std_mask = (weight <= 1.0).unsqueeze(1) * mask.float()
        
                # Calculate mean Huber loss per task head
                # Use 1e-8 for numerical stability in case a batch has 0 deep routes
                loss_deep = weighted_huber[deep_mask.bool()].mean() if deep_mask.sum() > 0 else torch.tensor(0.0, device=DEVICE)
                loss_std = weighted_huber[std_mask.bool()].mean() if std_mask.sum() > 0 else torch.tensor(0.0, device=DEVICE)
        
                # Combine with 0.5/0.5 weighting to prevent Standard routes from dominating
                # This forces the model to give equal importance to both play types
                loss_ball = (0.77 * loss_std) + (0.23 * loss_deep)
                loss = loss_ball / ACCUM_STEPS
        
                        # --- 5. MONITORING METRICS (RMSE) ---
                        # Calculate RMSE for logs, but do NOT backpropagate it
                with torch.no_grad():
                    # Calculate per-frame MSE for the entire batch
                    raw_mse = F.mse_loss(preds, target_expanded, reduction='none').mean(dim=-1)
                    
                    # Apply masks to get metrics for specific play types
                    # Average only over valid (unmasked) frames
                    current_rmse_deep = torch.sqrt((raw_mse * deep_mask).sum() / (deep_mask.sum() + 1e-8))
                    current_rmse_std = torch.sqrt((raw_mse * std_mask).sum() / (std_mask.sum() + 1e-8))
                    
                    # Global RMSE for the entire batch (standard monitoring)
                    current_rmse_global = torch.sqrt((raw_mse * mask.float()).sum() / (mask.sum() + 1e-8))
        
            # --- 6. BACKWARD PASS ---
            scaler.scale(loss).backward()
        
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
        
            # --- 7. ACCUMULATE LOGS ---
            total_ball_loss += loss_ball.item()
            total_ball_loss_rmse += current_rmse_global.item()
            
            batches += 1
        
        print(f"Epoch {epoch}/{EPOCHS} | "f"Avg Huber Loss={total_ball_loss/batches:.4f} | "f"Global RMSE={total_ball_loss_rmse/batches:.4f}")
        
        if epoch % CHECKPOINT_EVERY == 0:
            torch.save(model.state_dict(), f"checkpoints/model_epoch{epoch}.pt")
            print(f"💾 Saved checkpoint: model_epoch{epoch}.pt\n")

if __name__ == "__main__":
    
    FEATURES = ['off3_cb_x_early_initial', 'off3_lb_x_early_initial',
       'off4_lb_x_early_initial', 'off2_lb_x_early_initial',
       'off1_lb_x_early_initial', 'off3_s_x_early_initial',
       'off2_cb_x_early_initial', 'off3_nearest_def_x',
       'off2_nearest_def_x', 'off1_cb_x_early_initial',
       'off4_cb_x_early_initial', 'off4_nearest_def_x',
       'off1_nearest_def_x', 'def1_x', 'off2_s_x_early_initial',
       'off5_lb_x_early_initial', 'def4_x', 'off4_s_x_early_initial',
       'off1_s_x_early_initial', 'off1_pred_x_0.5s', 'endzone', 'def2_x',
       'off1_pred_x_1.0s', 'off1_x', 'off5_cb_x_early_initial', 'def3_x',
       'def5_x', 'def7_x', 'def6_x', 'off2_pred_x_1.0s', 'off4_x',
       'off5_x', 'off3_pred_x_1.0s', 'off5_s_x_early_initial',
       'off1_pred_x_2.0s', 'off2_x', 'off4_pred_x_2.0s',
       'off3_pred_x_2.0s', 'off3_x', 'off2_pred_x_2.0s',
       'off3_pred_x_3.0s', 'off2_pred_x_3.0s', 'off4_pred_x_3.0s',
       'off1_pred_x_3.0s', 'off3_wr_dist_endzone', 'endzone_bucket_idx',
       'off3_depth_offset', 'off2_wr_dist_endzone',
       'off1_wr_dist_endzone', 'off4_wr_dist_endzone',
       'off5_wr_dist_endzone', 'off1_depth_offset', 'off2_depth_offset',
       'off3_depth_leverage', 'CB_count', 'coverage_scheme_idx',
       'seconds_remaining_half_bucket_idx', 'off1_depth_leverage',
       'RB_count', 'receiver_alignment_idx', 'def1_def_qb_angle',
       'defenders_in_the_box', 'down', 'off2_depth_leverage', 'WR_count',
       'has_tempo', 'def7_def_qb_angle', 'TE_count',
       'offense_formation_idx', 'def8_def_qb_angle', 'def8_o', 'quarter',
       'play_direction_idx', 'DL_count', 'def8_def_qb_dist',
       'off1_boundary_pressure_x', 'def8_y',
       'coverage_responsibility_idx', 'off5_wr_qb_dist',
       'off2_leverage_angle', 'off5_wr_endzone_ratio',
       'off5_boundary_pressure_x', 'off3_wr_qb_dist', 'yards_to_go',
       'off3_path_angle_cos', 'off5_path_angle_delta',
       'off5_path_angle_sin', 'off3_leverage_angle', 'def6_def_qb_angle',
       'off2_boundary_pressure_x', 'def3_def_qb_angle',
       'off3_wr_endzone_ratio', 'def2_def_qb_angle', 'qb_dropback_masked',
       'def7_def_qb_dist', 'def4_def_qb_dist',
       'off1_wr_qb_dist', 'off2_wr_qb_dist', 'off4_wr_qb_dist',
       'off4_leverage_angle', 'off1_wr_endzone_ratio',
       'off4_wr_endzone_ratio', 'def5_def_qb_angle',
       'off2_wr_endzone_ratio', 'off5_path_angle_cos', 'def1_def_qb_dist',
       'off3_boundary_pressure_x', 'def5_def_qb_dist',
       'off3_path_angle_sin', 'def3_def_qb_dist', 'def2_def_qb_dist',
       'off4_boundary_pressure_x', 'off4_path_angle_sin',
       'off4_path_angle_cos', 'def4_def_qb_angle',
       'off4_path_angle_delta', 'qb_name_idx', 'off3_path_angle_delta',
       'off2_path_angle_cos', 'off2_path_angle_sin', 'point_differential',
       'off1_path_angle_delta', 'possession_team_idx',
       'off1_path_angle_cos', 'off1_path_angle_sin',
       'off2_path_angle_delta', 'off3_s_depth_initial',
       'off5_cb_orient_early', 'off5_s_depth_initial',
       'off3_cb_depth_initial', 'off5_cb_depth_initial',
       'gameClock_seconds', 'off5_v_x', 'defensive_team_idx',
       'off5_boundary_pressure_y', 'off5_lb_orient_early_initial',
       'off5_lb_y_early_initial',
       'off5_triangle_depth_variance_early_initial',
       'off5_cb_orient_early_initial', 'off5_dir',
       'off5_cb_y_early_initial', 'qb_qb_y_xdelta_5',
       'off5_triangle_perimeter_early_initial', 'off5_nearest_def_y',
       'off5_pred_y_1.0s', 'off5_o',
       'off3_triangle_depth_variance_early_initial',
        'off5_cb_dist_early_initial', 'off5_pred_y_0.5s', 'off1_wr_dist_to_sideline',
         'off2_wr_dist_to_sideline',
         'off3_wr_dist_to_sideline',
         'off4_wr_dist_to_sideline',
         'off5_wr_dist_to_sideline',
         'off1_wr_eye_contact_angle',
         'off2_wr_eye_contact_angle',
         'off3_wr_eye_contact_angle',
         'off4_wr_eye_contact_angle',
         'off5_wr_eye_contact_angle',
         'off1_target_nearest_sideline_left_idx',
         'off2_target_nearest_sideline_left_idx',
         'off3_target_nearest_sideline_left_idx',
         'off4_target_nearest_sideline_left_idx',
         'off5_target_nearest_sideline_left_idx', 'qb_qb_orientation', 'qb_qb_orientation_delta10', 'off1_speed_change', 'off2_speed_change', 
                                                    'off3_speed_change', 'off4_speed_change', 'off5_speed_change', 'qb_pressure_index', 'off1_alignment_idx', 'off2_alignment_idx', 'off3_alignment_idx', 'off4_alignment_idx', 'off5_alignment_idx', 'def1_alignment_idx', 'def2_alignment_idx', 'def3_alignment_idx', 'def4_alignment_idx', 'def5_alignment_idx', 'def6_alignment_idx', 'def7_alignment_idx', 'def8_alignment_idx', 'def9_alignment_idx', 'def10_alignment_idx', 'def11_alignment_idx']
    
    main(FEATURES)

