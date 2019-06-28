#!/bin/bash


python ./experiments/language_script.py --K 10 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 0 

python ./experiments/language_script.py --K 10 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 1 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 1 

python ./experiments/language_script.py --K 10 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 2 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 2 

python ./experiments/language_script.py --K 10 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 3 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 3 

python ./experiments/language_script.py --K 10 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 4 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 4 

python ./experiments/language_script.py --K 10 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 5 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 5 

python ./experiments/language_script.py --K 10 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 6 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 6 

python ./experiments/language_script.py --K 10 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 7 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 7 

python ./experiments/language_script.py --K 10 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 8 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 8 

python ./experiments/language_script.py --K 10 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 9 

python ./experiments/language_script.py --K 10 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 1 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 10 

python ./experiments/language_script.py --K 10 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 2 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 11 

python ./experiments/language_script.py --K 10 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 3 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 12 

python ./experiments/language_script.py --K 10 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 4 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 13 

python ./experiments/language_script.py --K 10 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 5 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 14 

python ./experiments/language_script.py --K 10 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 6 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 15 

python ./experiments/language_script.py --K 10 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 7 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 16 

python ./experiments/language_script.py --K 10 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 8 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 17 

python ./experiments/language_script.py --K 50 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 18 

python ./experiments/language_script.py --K 50 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 1 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 19 

python ./experiments/language_script.py --K 50 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 2 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 20 

python ./experiments/language_script.py --K 50 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 3 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 21 

python ./experiments/language_script.py --K 50 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 4 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 22 

python ./experiments/language_script.py --K 50 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 5 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 23 

python ./experiments/language_script.py --K 50 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 6 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 24 

python ./experiments/language_script.py --K 50 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 7 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 25 

python ./experiments/language_script.py --K 50 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 8 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 26 

python ./experiments/language_script.py --K 50 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 27 

python ./experiments/language_script.py --K 50 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 1 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 28 

python ./experiments/language_script.py --K 50 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 2 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 29 

python ./experiments/language_script.py --K 50 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 3 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 30 

python ./experiments/language_script.py --K 50 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 4 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 31 

python ./experiments/language_script.py --K 50 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 5 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 32 

python ./experiments/language_script.py --K 50 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 6 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 33 

python ./experiments/language_script.py --K 50 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 7 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 34 

python ./experiments/language_script.py --K 50 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 8 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 35 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.9 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 36 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.9 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 1 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 37 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.9 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 2 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 38 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.9 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 3 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 39 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.9 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 4 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 40 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.9 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 5 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 41 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.9 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 6 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 42 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.9 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 7 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 43 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.9 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 8 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 44 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.9 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 45 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.9 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 1 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 46 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.9 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 2 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 47 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.9 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 3 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 48 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.9 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 4 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 49 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.9 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 5 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 50 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.9 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 6 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 51 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.9 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 7 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 52 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.9 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 8 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 53 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.5 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 54 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.5 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 1 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 55 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.5 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 2 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 56 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.5 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 3 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 57 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.5 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 4 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 58 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.5 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 5 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 59 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.5 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 6 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 60 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.5 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 7 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 61 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.5 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 8 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 62 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.5 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 63 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.5 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 1 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 64 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.5 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 2 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 65 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.5 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 3 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 66 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.5 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 4 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 67 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.5 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 5 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 68 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.5 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 6 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 69 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.5 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 7 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 70 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.5 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 8 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 71 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.1 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 72 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.1 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 1 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 73 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.1 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 2 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 74 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.1 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 3 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 75 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.1 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 4 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 76 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.1 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 5 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 77 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.1 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 6 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 78 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.1 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 7 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 79 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.1 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 8 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 80 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.1 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 81 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.1 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 1 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 82 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.1 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 2 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 83 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.1 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 3 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 84 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.1 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 4 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 85 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.1 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 5 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 86 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.1 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 6 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 87 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.1 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 7 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 88 

python ./experiments/language_script.py --K 100 --adaptive_K --batch_size 32 --beta_estimate_method ols --cuda --data ./data/ptb --delta 0.1 --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 8 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 89 

python ./experiments/language_script.py --K 100 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 90 

python ./experiments/language_script.py --K 100 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 1 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 91 

python ./experiments/language_script.py --K 100 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 2 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 92 

python ./experiments/language_script.py --K 100 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 3 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 93 

python ./experiments/language_script.py --K 100 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 4 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 94 

python ./experiments/language_script.py --K 100 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 5 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 95 

python ./experiments/language_script.py --K 100 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 6 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 96 

python ./experiments/language_script.py --K 100 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 7 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 97 

python ./experiments/language_script.py --K 100 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 8 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 98 

python ./experiments/language_script.py --K 100 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 99 

python ./experiments/language_script.py --K 100 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 1 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 100 

python ./experiments/language_script.py --K 100 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 2 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 101 

python ./experiments/language_script.py --K 100 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 3 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 102 

python ./experiments/language_script.py --K 100 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 4 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 103 

python ./experiments/language_script.py --K 100 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 5 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 104 

python ./experiments/language_script.py --K 100 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 6 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 105 

python ./experiments/language_script.py --K 100 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 7 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 106 

python ./experiments/language_script.py --K 100 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 8 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 107 

python ./experiments/language_script.py --K 200 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 108 

python ./experiments/language_script.py --K 200 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 1 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 109 

python ./experiments/language_script.py --K 200 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 2 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 110 

python ./experiments/language_script.py --K 200 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 3 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 111 

python ./experiments/language_script.py --K 200 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 4 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 112 

python ./experiments/language_script.py --K 200 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 5 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 113 

python ./experiments/language_script.py --K 200 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 6 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 114 

python ./experiments/language_script.py --K 200 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 7 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 115 

python ./experiments/language_script.py --K 200 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 8 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 116 

python ./experiments/language_script.py --K 200 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 117 

python ./experiments/language_script.py --K 200 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 1 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 118 

python ./experiments/language_script.py --K 200 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 2 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 119 

python ./experiments/language_script.py --K 200 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 3 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 120 

python ./experiments/language_script.py --K 200 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 4 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 121 

python ./experiments/language_script.py --K 200 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 5 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 122 

python ./experiments/language_script.py --K 200 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 6 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 123 

python ./experiments/language_script.py --K 200 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 7 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 124 

python ./experiments/language_script.py --K 200 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 8 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 125 

python ./experiments/language_script.py --K 300 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 126 

python ./experiments/language_script.py --K 300 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 1 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 127 

python ./experiments/language_script.py --K 300 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 2 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 128 

python ./experiments/language_script.py --K 300 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 3 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 129 

python ./experiments/language_script.py --K 300 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 4 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 130 

python ./experiments/language_script.py --K 300 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 5 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 131 

python ./experiments/language_script.py --K 300 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 6 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 132 

python ./experiments/language_script.py --K 300 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 7 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 133 

python ./experiments/language_script.py --K 300 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 8 --lr 10.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 134 

python ./experiments/language_script.py --K 300 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 0 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 135 

python ./experiments/language_script.py --K 300 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 1 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 136 

python ./experiments/language_script.py --K 300 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 2 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 137 

python ./experiments/language_script.py --K 300 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 3 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 138 

python ./experiments/language_script.py --K 300 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 4 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 139 

python ./experiments/language_script.py --K 300 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 5 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 140 

python ./experiments/language_script.py --K 300 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 6 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 141 

python ./experiments/language_script.py --K 300 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 7 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 142 

python ./experiments/language_script.py --K 300 --batch_size 32 --cuda --data ./data/ptb --dropout 0.2 --emsize 900 --experiment_folder ./output/ptb-lstm --init_num 8 --lr 1.0 --model LSTM --nhid 900 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 143 

python ./experiments/aggregate_output.py --path_to_data ./output/ptb-lstm
#EOF