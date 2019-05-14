#!/bin/bash


python ./experiments/language_script.py --K 10 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 0 

python ./experiments/language_script.py --K 10 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 1 

python ./experiments/language_script.py --K 50 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 2 

python ./experiments/language_script.py --K 50 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 3 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.9 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 4 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.9 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 5 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.5 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 6 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.5 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 7 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.1 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 8 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.1 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 9 

python ./experiments/language_script.py --K 100 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 10 

python ./experiments/language_script.py --K 100 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 11 

python ./experiments/language_script.py --K 200 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 12 

python ./experiments/language_script.py --K 200 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 13 

python ./experiments/language_script.py --K 300 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 14 

python ./experiments/language_script.py --K 300 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 15 

python ./experiments/aggregate_output.py --path_to_data ./output/ptb-lstm
#EOF