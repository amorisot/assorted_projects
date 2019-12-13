The capsule network reproduction. It was done for a group project in the class in my practical machine learning M.Sc. course. 
You shouldn't be able to run it, because it assumes that the scripts are being run on the Edinburgh informatics cluster, which have access to pre-downloaded and formatted academic datasets like smallnorb. The report is good though, it got a 90!

## G63-MLP
Repo for team G63 in MLP

## Idea

Check this repo out locally and on the GPU cluster.

1. Set up a config (local machine)

Each experiment should have a different config file (see `scripts/runners/config`). These have things that can affect the experiment like architecture and early stopping configs.
An experiment could be "capsnets on cifar10" or "cnn arch 1 on mnist".

I'd debug this locally. To make sure it can build the graph, use

    python run_from_config.py --network-config config/[config_name.yaml] --debug
    
Then double check that it can run a minibatch. This can catch errors like the input batch being the wrong size. It also generates tensorflow graphs that you can use to verify. The output of the run gives a command to view the tensorgraph.

    python run_from_config.py --network-config config/[config_name.yaml] --graph-only --output-location [path to folder]

Then commit the yaml file, and push it.

2. Run the job on the GPU cluster

Checkout or pull this repo on the GPU cluster and run something like

    cd G63-MLP/scripts/runners
    sbatch run_from_config_on_gpu.sh config/sample_capsnet.yaml

Don't forget the config. Verify the job is running using `squeue`.

This will start a job using this config with a `run_id` based on the timestamp. A run is uniquely defined by its
config_id and run_id.

3. View results

The job will save a bunch of files in `~/experiment_results`. An easy way to view them is

    python explore_experiment_results.py recent
    
At the bottom of the output is another command to view details of a specific run.

3. Save results (GPU cluster)

To "commit" results, move over the stats file. The `recent` command can also print `cp` statements.

    python explore_experiment_results.py recent --commit-help


## Notebooks

The easiest way to use this repo locally is to set up the mlp environment and run the notebooks in there. The instructions to view these notebooks are approximately:

 - Set up the [mlp environment](https://github.com/CSTR-Edinburgh/mlpractical/blob/mlp2017-8/semester_2_materials/notes/environment-set-up.md)
 - `source activate mlp`
 - and run `jupyter notebook` in this repos' directory.

There are a few notebooks in `notebooks/`.


## Bonus: Tests

Install pytest

    pip install -r requirements.txt

Then to run the tests do

    cd scripts/runners
    PYTHONPATH=. pytest tests/test_runner.py

If you want to print output of non-failing tests

    scripts/runners
    PYTHONPATH=. pytest tests/test_runner.py -s


## Restoring

On the GPU, the easiest way to restore is to run `recent` and follow the advice at the bottom of the page.

    python explore_experiment_results.py recent

For example

    sbatch restore_run_from_config_on_gpu.sh /home/s1164250/experiment_results/capsnet_smallnorb_leaky_relu_learning_rate_0_0001_dropout/1520444648
    
Locally, you can run

    python run_from_config.py --restore-from-location /home/s1164250/experiment_results/capsnet_smallnorb_leaky_relu_learning_rate_0_0001_dropout/1520444648

It tries to look up the proper config. If the config isn't in the normal place, use the `--restore-from-location-config-folder` flag.



