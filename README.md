# Adaptive Truncated Backpropagation Through Time
> This is PyTorch code for Adaptive TBPTT used for training RNN

This repo contains the python code for the experiments in [Adaptively Truncating Backpropagation Through Time to Control Gradient Bias](https://arxiv.org/abs/1905.07473) by Christopher Aicher, Nicholas J. Foti, and Emily B. Fox. UAI 2019.

## Overview
* The `data` folder stores the synthetic, language model, and temporal point process data.
* The `scripts` folder stores the bash scripts for the experiments.
* The `experiments` folder contains the python code for experiment scripts. See `experiments/README.md` for additional details.
* The `tbptt` folder contains the python module code. See `tbptt/README.md` for additional details.

## Installation
Add the `tbptt` folder to the PYTHONPATH.

Requirements:
Python 3+, pytorch, numpy, pandas, scipy, matplotlib, seaborn, joblib, scikit-learn, tqdm

## Usage example
To download the language modeling text data call `get_lm_data.sh`.
```
cd <path_to_repo>
./get_lm_data
```

To download the temporal point process data call `get_tpp_data.sh`.
```
cd <path_to_repo>
./get_tpp_data
```

To run the experiment scripts, call individual experiment bash scripts
```
cd <path_to_repo>
./scripts/<experiment_name>.sh
```
This may take a while depending on your setup.

The output of each experiment is a pair files (`options.csv` and `metric.csv.gz`) at `./output/<experiment_name>/out/`.
* `options.csv` is a CSV with a row for each TBPTT setup (i.e. for `K = 10, 50, 100` or `lr = 1, 10`, etc.) and columns with training setup values (i.e. `lr`, `K`, `batchsize`, etc.).
* `metrics.csv.gz` is a compressed CSV with a row for each measured `epoch` and `metric` (i.e. `valid_ppl`, `test_ppl`, `logdelta`, etc.)

You can join `options.csv` with `metrics.csv` on the `experiment_id` column.
See `experiments/README.md` for additional details.

**The default bash scripts require the use of PyTorch with a GPU**.
If you do not have a GPU, set `CUDA_AVAILABLE` to `False` in `./experiments/script_maker.py` and regenerate the bash scripts
```
cd <path_to_repo>
python ./experiments/script_maker.py
```


## Release History / Changelog

* 0.2.0
    * The second release (July 2019)
    * Added temporal point process modeling
    * Updated scripts to use multiple initializations
* 0.1.0
    * The first release (May 2019)

## Meta

Christopher Aicher – aicherc@uw.edu

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/aicherc/adaptive_tbptt](https://github.com/aicherc/adaptive_tbptt)

## Contributing

1. Fork it (<https://github.com/aicherc/adaptive_tbptt/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

<!-- Markdown link & img dfn's -->
<!-- Goes HERE -->
