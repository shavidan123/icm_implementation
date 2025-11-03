# Implementation of Internal Coherence Maximization Algorithm 1 on TruthfulQA

How to reproduce:

**Step 1:** Create virtual environment and install dependencies
Input the following commands (or set up an environment some other way)
python -m venv .venv #or python3 -m venv .venv
source .venv/bin/activate #or source .venv/Scripts/activate (i used windows for this)
pip install -r requirements.txt

OR conda should work as well
conda create -n icm-qa python=3.11 -y
conda activate icm-qa
pip install -r requirements.txt

**Step 2:** Set the api key in your environment
You can do this with the following commands:
export HYPERBOLIC_API_KEY="...your-key..."
or $env:HYPERBOLIC_API_KEY="...your-key..." (windows)
OR set in config (worst case)
I allowed you to set in the config for simplicity but this is bad api practice so don't do it

**Step 3:** Configure the parameters of the run by modifying the config file. I wasn't sure the best way to do this, but it seemed to work well enough for the purposes of this experiment. It's mostly for tuning the ICM hyperparameters. I've currently just set them to the default ones mentioned in the paper.

**Step 4:** Run the code and see the results

You can run the code by running the "run_icm" python file. This can be done from the icm_implementation directory with the command:

python -m scripts.run_icm

Thank you!