#!/bin/bash


python ./experiments/language_script.py --K 10 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 0 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 0 

python ./experiments/language_script.py --K 50 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 0 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 1 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.9 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 0 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 2 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.5 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 0 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 3 

python ./experiments/language_script.py --K 100 --adaptive_K --beta_estimate_method ols --cuda --data ./data/wikitext-2 --delta 0.1 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 0 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 4 

python ./experiments/language_script.py --K 100 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 0 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 5 

python ./experiments/language_script.py --K 200 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 0 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 6 

python ./experiments/language_script.py --K 300 --cuda --data ./data/wikitext-2 --dropout 0.2 --emsize 512 --epoch 50 --experiment_folder ./output/wiki2-lstm --init_num 0 --linear_scale --lr 10.0 --max_train_time 10800 --model LSTM --nhid 512 --nlayers 1 --scale_lr_K --tbptt_style original-buffer --tied --weight_decay 1e-07 --experiment_id 7 

python ./experiments/aggregate_output.py --path_to_data ./output/wiki2-lstm
#EOF