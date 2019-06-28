#!/bin/bash


python ./experiments/temporal_pp_script.py --K 3 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 0 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 0 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.9 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 0 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 1 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.5 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 0 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 2 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 0 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 3 

python ./experiments/temporal_pp_script.py --K 6 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 0 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 4 

python ./experiments/temporal_pp_script.py --K 9 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 0 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 5 

python ./experiments/temporal_pp_script.py --K 15 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 0 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 6 

python ./experiments/temporal_pp_script.py --K 21 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 0 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 7 

python ./experiments/temporal_pp_script.py --K 3 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 1 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 8 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.9 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 1 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 9 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.5 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 1 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 10 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 1 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 11 

python ./experiments/temporal_pp_script.py --K 6 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 1 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 12 

python ./experiments/temporal_pp_script.py --K 9 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 1 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 13 

python ./experiments/temporal_pp_script.py --K 15 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 1 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 14 

python ./experiments/temporal_pp_script.py --K 21 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 1 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 15 

python ./experiments/temporal_pp_script.py --K 3 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 2 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 16 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.9 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 2 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 17 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.5 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 2 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 18 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 2 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 19 

python ./experiments/temporal_pp_script.py --K 6 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 2 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 20 

python ./experiments/temporal_pp_script.py --K 9 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 2 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 21 

python ./experiments/temporal_pp_script.py --K 15 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 2 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 22 

python ./experiments/temporal_pp_script.py --K 21 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 2 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 23 

python ./experiments/temporal_pp_script.py --K 3 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 3 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 24 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.9 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 3 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 25 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.5 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 3 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 26 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 3 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 27 

python ./experiments/temporal_pp_script.py --K 6 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 3 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 28 

python ./experiments/temporal_pp_script.py --K 9 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 3 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 29 

python ./experiments/temporal_pp_script.py --K 15 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 3 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 30 

python ./experiments/temporal_pp_script.py --K 21 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 3 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 31 

python ./experiments/temporal_pp_script.py --K 3 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 4 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 32 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.9 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 4 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 33 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.5 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 4 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 34 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 4 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 35 

python ./experiments/temporal_pp_script.py --K 6 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 4 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 36 

python ./experiments/temporal_pp_script.py --K 9 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 4 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 37 

python ./experiments/temporal_pp_script.py --K 15 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 4 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 38 

python ./experiments/temporal_pp_script.py --K 21 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 4 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 39 

python ./experiments/temporal_pp_script.py --K 3 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 5 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 40 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.9 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 5 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 41 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.5 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 5 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 42 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 5 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 43 

python ./experiments/temporal_pp_script.py --K 6 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 5 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 44 

python ./experiments/temporal_pp_script.py --K 9 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 5 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 45 

python ./experiments/temporal_pp_script.py --K 15 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 5 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 46 

python ./experiments/temporal_pp_script.py --K 21 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 5 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 47 

python ./experiments/temporal_pp_script.py --K 3 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 6 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 48 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.9 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 6 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 49 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.5 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 6 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 50 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 6 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 51 

python ./experiments/temporal_pp_script.py --K 6 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 6 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 52 

python ./experiments/temporal_pp_script.py --K 9 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 6 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 53 

python ./experiments/temporal_pp_script.py --K 15 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 6 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 54 

python ./experiments/temporal_pp_script.py --K 21 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 6 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 55 

python ./experiments/temporal_pp_script.py --K 3 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 7 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 56 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.9 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 7 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 57 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.5 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 7 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 58 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 7 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 59 

python ./experiments/temporal_pp_script.py --K 6 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 7 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 60 

python ./experiments/temporal_pp_script.py --K 9 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 7 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 61 

python ./experiments/temporal_pp_script.py --K 15 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 7 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 62 

python ./experiments/temporal_pp_script.py --K 21 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 7 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 63 

python ./experiments/temporal_pp_script.py --K 3 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 8 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 64 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.9 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 8 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 65 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.5 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 8 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 66 

python ./experiments/temporal_pp_script.py --K 6 --adaptive_K --beta_estimate_method ols --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --delta 0.1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 8 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --scale_lr_K --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 67 

python ./experiments/temporal_pp_script.py --K 6 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 8 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 68 

python ./experiments/temporal_pp_script.py --K 9 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 8 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 69 

python ./experiments/temporal_pp_script.py --K 15 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 8 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 70 

python ./experiments/temporal_pp_script.py --K 21 --cuda --data_path ./data/tpp-book-order --data_split_number 1 --data_time_scale 1 --dropout 0.0 --emsize 128 --epochs 51 --experiment_folder ./output/book_order-lstm --init_num 8 --lr 0.1 --model LSTM --nhid 128 --nlayers 2 --optim SGD --tbptt_style original-buffer --time_onehot_split --weight_decay 0.001 --experiment_id 71 

python ./experiments/aggregate_output.py --path_to_data ./output/book_order-lstm
#EOF