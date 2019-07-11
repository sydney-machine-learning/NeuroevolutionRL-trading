import argparse
import os, sys
from datetime import datetime
import multiprocessing as mp
import yaml


def get_runid(_datetime):
    """
    Generate run_id from datetime object

    Parameters
    ----------
    _datetime: datetime object
        Current Date and Time

    Returns
    -------
    run_id: str
        run_id consisting of date and time
    """
    date, time = str(_datetime).split()
    time = time.split('.')[0]
    run_id = "_".join([date, time])
    return run_id

# CREATE ARGUMENT PARSER
parser = argparse.ArgumentParser(description='Run Evolutionary Parallel Tempering')

# ADD ARGUMENTS TO THE PARSER
parser.add_argument('--problem', type=str, default='synthetic', help='Problem to be used for Evolutionary PT: \n"synthetic", "iris", "ions", "cancer", "bank", "penDigit", "chess", "TicTacToe')
parser.add_argument('--num-chains', type=int, default=mp.cpu_count()-2, help='Number of Chains for Parallel Tempering')
parser.add_argument('--population-size', type=int, default=200, help='Population size for G3PCX Evolutionary algorithm.')
parser.add_argument('--num-samples', type=int, default=None, help='Total number of samples (all chains combined).')
parser.add_argument('--swap-interval', type=int, default=10, help='Number of samples between each swap.')
parser.add_argument('--burn-in', type=float, default=0.2, help='Ratio of samples to be discarded as burn-in samples. Value 0.1 means 10 percent samples are discarded')
parser.add_argument('--max-temp', type=float, default=5, help='Temperature to be assigned to the chain with maximum temperature.')
parser.add_argument('--topology', type=str, default=None, help='String representation of network topology. Eg:- "input,hidden,output"')
parser.add_argument('--run-id', type=str, default=get_runid(datetime.now())
, help="Unique Id to identify run.")
parser.add_argument('--root', type=str, default=os.path.split(os.getcwd())[0], help="path to root directory (evolutionary-pt).")
parser.add_argument('--train-data', type=str, default=None, help='Path to the train data')
parser.add_argument('--test-data', type=str, default=None, help='Path to the test data')
parser.add_argument('--config-file', type=str, default=None, help='Path to data config yaml file')


# PARSE COMMANDLINE ARGUMENTS
opt = parser.parse_args()

# CHECK FOR CONFIG YAML PATH
if opt.config_file is None:
    opt.config_file = 'data.yaml'

# READ CONFIG DATA CONFIG FROM YAML FILE
with open(opt.config_file, 'r') as stream:
    data_config = yaml.safe_load(stream)[opt.problem]

# UPDATE NONE ARGUMENTS
if opt.train_data is None:
    opt.train_data = data_config['train_data']
if opt.test_data is None:
    opt.test_data = data_config['test_data']
if opt.num_samples is None:
    opt.num_samples = data_config['num_samples']
if opt.topology is None:
    opt.topology = data_config['topology']

# ADD REMAINING ARGS FROM CONFIG
opt.problem_name = data_config['name']
opt.problem_type = data_config['problem_type']