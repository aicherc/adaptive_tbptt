#!/bin/bash


python ./experiments/language_script.py --K 10 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 0 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 0 

python ./experiments/language_script.py --K 10 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 1 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 1 

python ./experiments/language_script.py --K 10 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 2 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 2 

python ./experiments/language_script.py --K 10 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 3 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 3 

python ./experiments/language_script.py --K 10 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 4 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 4 

python ./experiments/language_script.py --K 10 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 5 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 5 

python ./experiments/language_script.py --K 10 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 6 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 6 

python ./experiments/language_script.py --K 10 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 7 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 7 

python ./experiments/language_script.py --K 10 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 8 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 8 

python ./experiments/language_script.py --K 50 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 0 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 9 

python ./experiments/language_script.py --K 50 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 1 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 10 

python ./experiments/language_script.py --K 50 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 2 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 11 

python ./experiments/language_script.py --K 50 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 3 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 12 

python ./experiments/language_script.py --K 50 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 4 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 13 

python ./experiments/language_script.py --K 50 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 5 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 14 

python ./experiments/language_script.py --K 50 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 6 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 15 

python ./experiments/language_script.py --K 50 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 7 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 16 

python ./experiments/language_script.py --K 50 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 8 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 17 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.9 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 0 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 18 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.9 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 1 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 19 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.9 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 2 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 20 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.9 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 3 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 21 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.9 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 4 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 22 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.9 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 5 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 23 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.9 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 6 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 24 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.9 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 7 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 25 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.9 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 8 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 26 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.5 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 0 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 27 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.5 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 1 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 28 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.5 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 2 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 29 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.5 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 3 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 30 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.5 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 4 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 31 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.5 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 5 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 32 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.5 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 6 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 33 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.5 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 7 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 34 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.5 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 8 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 35 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.1 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 0 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 36 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.1 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 1 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 37 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.1 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 2 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 38 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.1 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 3 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 39 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.1 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 4 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 40 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.1 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 5 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 41 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.1 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 6 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 42 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.1 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 7 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 43 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.1 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 8 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 44 

python ./experiments/language_script.py --K 100 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 0 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 45 

python ./experiments/language_script.py --K 100 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 1 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 46 

python ./experiments/language_script.py --K 100 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 2 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 47 

python ./experiments/language_script.py --K 100 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 3 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 48 

python ./experiments/language_script.py --K 100 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 4 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 49 

python ./experiments/language_script.py --K 100 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 5 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 50 

python ./experiments/language_script.py --K 100 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 6 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 51 

python ./experiments/language_script.py --K 100 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 7 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 52 

python ./experiments/language_script.py --K 100 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 8 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 53 

python ./experiments/language_script.py --K 200 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 0 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 54 

python ./experiments/language_script.py --K 200 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 1 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 55 

python ./experiments/language_script.py --K 200 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 2 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 56 

python ./experiments/language_script.py --K 200 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 3 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 57 

python ./experiments/language_script.py --K 200 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 4 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 58 

python ./experiments/language_script.py --K 200 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 5 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 59 

python ./experiments/language_script.py --K 200 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 6 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 60 

python ./experiments/language_script.py --K 200 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 7 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 61 

python ./experiments/language_script.py --K 200 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 8 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 62 

python ./experiments/language_script.py --K 300 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 0 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 63 

python ./experiments/language_script.py --K 300 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 1 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 64 

python ./experiments/language_script.py --K 300 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 2 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 65 

python ./experiments/language_script.py --K 300 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 3 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 66 

python ./experiments/language_script.py --K 300 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 4 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 67 

python ./experiments/language_script.py --K 300 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 5 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 68 

python ./experiments/language_script.py --K 300 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 6 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 69 

python ./experiments/language_script.py --K 300 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 7 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 70 

python ./experiments/language_script.py --K 300 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 8 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 71 

python ./experiments/aggregate_output.py --path_to_data ./output/wiki2-lstm
#EOF