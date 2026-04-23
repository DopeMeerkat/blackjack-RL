@echo off
:: Get timestamp in YYYYMMDD_HHMMSS format
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%

call "E:\Miniconda3\Scripts\activate.bat" "E:\Miniconda3\envs\rl_proj"

python train.py ^
    --total_steps   5000000 ^
    --rollout_steps 2048    ^
    --n_envs        4       ^
    --batch_size    512     ^
    --n_epochs      10      ^
    --lr            2e-4    ^
    --aux_lr        1e-3    ^
    --gamma         0.99    ^
    --gae_lambda    0.95    ^
    --clip_eps      0.2     ^
    --ent_coef      0.05    ^
    --vf_bet_coef   0.5     ^
    --vf_play_coef  0.5     ^
    --max_grad_norm 0.5     ^
    --target_kl     0.02    ^
    --d_shoe        128     ^
    --d_hand        32      ^
    --d_fused       192     ^
    --stage1_end    500000  ^
    --stage2_end    2500000 ^
    --aux_anneal    500000  ^
    --flat_bet      10.0    ^
    --no_bet_curriculum     ^
    --seed          43      ^
    --device        cuda    ^
    --log_dir       "runs\%TIMESTAMP%" ^
    --save_dir      "checkpoints\%TIMESTAMP%"    ^
    --save_every    250000  ^
    --log_every     5

cmd /k