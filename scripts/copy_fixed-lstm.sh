#!/bin/bash


python ./experiments/synth_script.py --K 5 --cuda --data_lag 10 --data_minlag 10 --data_name ./data/copy_fixed --data_type copy --dropout 0.0 --emsize 6 --experiment_folder ./output/copy_fixed-lstm --init_num 0 --lr 1.0 --model LSTM --nhid 50 --nlayers 2 --scale_lr_K --tbptt_style original-buffer --weight_decay 1e-05 --experiment_id 0 

python ./experiments/synth_script.py --K 10 --cuda --data_lag 10 --data_minlag 10 --data_name ./data/copy_fixed --data_type copy --dropout 0.0 --emsize 6 --experiment_folder ./output/copy_fixed-lstm --init_num 0 --lr 1.0 --model LSTM --nhid 50 --nlayers 2 --scale_lr_K --tbptt_style original-buffer --weight_decay 1e-05 --experiment_id 1 

python ./experiments/synth_script.py --K 15 --adaptive_K --beta_estimate_method ols --cuda --data_lag 10 --data_minlag 10 --data_name ./data/copy_fixed --data_type copy --delta 0.9 --dropout 0.0 --emsize 6 --experiment_folder ./output/copy_fixed-lstm --init_num 0 --lr 1.0 --model LSTM --nhid 50 --nlayers 2 --scale_lr_K --tbptt_style original-buffer --weight_decay 1e-05 --experiment_id 2 

python ./experiments/synth_script.py --K 15 --adaptive_K --beta_estimate_method ols --cuda --data_lag 10 --data_minlag 10 --data_name ./data/copy_fixed --data_type copy --delta 0.5 --dropout 0.0 --emsize 6 --experiment_folder ./output/copy_fixed-lstm --init_num 0 --lr 1.0 --model LSTM --nhid 50 --nlayers 2 --scale_lr_K --tbptt_style original-buffer --weight_decay 1e-05 --experiment_id 3 

python ./experiments/synth_script.py --K 15 --adaptive_K --beta_estimate_method ols --cuda --data_lag 10 --data_minlag 10 --data_name ./data/copy_fixed --data_type copy --delta 0.1 --dropout 0.0 --emsize 6 --experiment_folder ./output/copy_fixed-lstm --init_num 0 --lr 1.0 --model LSTM --nhid 50 --nlayers 2 --scale_lr_K --tbptt_style original-buffer --weight_decay 1e-05 --experiment_id 4 

python ./experiments/synth_script.py --K 15 --cuda --data_lag 10 --data_minlag 10 --data_name ./data/copy_fixed --data_type copy --dropout 0.0 --emsize 6 --experiment_folder ./output/copy_fixed-lstm --init_num 0 --lr 1.0 --model LSTM --nhid 50 --nlayers 2 --scale_lr_K --tbptt_style original-buffer --weight_decay 1e-05 --experiment_id 5 

python ./experiments/synth_script.py --K 20 --cuda --data_lag 10 --data_minlag 10 --data_name ./data/copy_fixed --data_type copy --dropout 0.0 --emsize 6 --experiment_folder ./output/copy_fixed-lstm --init_num 0 --lr 1.0 --model LSTM --nhid 50 --nlayers 2 --scale_lr_K --tbptt_style original-buffer --weight_decay 1e-05 --experiment_id 6 

python ./experiments/synth_script.py --K 30 --cuda --data_lag 10 --data_minlag 10 --data_name ./data/copy_fixed --data_type copy --dropout 0.0 --emsize 6 --experiment_folder ./output/copy_fixed-lstm --init_num 0 --lr 1.0 --model LSTM --nhid 50 --nlayers 2 --scale_lr_K --tbptt_style original-buffer --weight_decay 1e-05 --experiment_id 7 

python ./experiments/aggregate_output.py --path_to_data ./output/copy_fixed-lstm
#EOF