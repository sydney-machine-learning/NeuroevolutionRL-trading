import argparse
import os
from datetime import datetime
import multiprocessing as mp


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
#parser.add_argument('--problem', type=str, default='synthetic', help='Problem to be used for Evolutionary PT: \n"synthetic", "iris", "ions", "cancer", "bank", "penDigit", "chess", "TicTacToe')
parser.add_argument('--num-chains', type=int, default=mp.cpu_count()-2, help='Number of Chains for Parallel Tempering')
parser.add_argument('--population-size', type=int, default=200, help='Population size for G3PCX Evolutionary algorithm.')
parser.add_argument('--num-samples', type=int, default=None, help='Total number of samples (all chains combined).')
parser.add_argument('--swap-interval', type=int, default=10, help='Number of samples between each swap.')
parser.add_argument('--run-id', type=str, default=get_runid(datetime.now())
, help="Unique Id to identify run.")
parser.add_argument('--root', type=str, default=os.path.split(os.getcwd())[0], help="path to root directory (Evo_algo_RL).")

#parser.add_argument('--burn-in', type=float, default=0.2, help='Ratio of samples to be discarded as burn-in samples. Value 0.1 means 10 percent samples are discarded')
#parser.add_argument('--max-temp', type=float, default=5, help='Temperature to be assigned to the chain with maximum temperature.')
#parser.add_argument('--topology', type=str, default=None, help='String representation of network topology. Eg:- "input,hidden,output"')

#parser.add_argument('--train-data', type=str, default=None, help='Path to the train data')
#parser.add_argument('--test-data', type=str, default=None, help='Path to the test data')
#parser.add_argument('--config-file', type=str, default=None, help='Path to data config yaml file')


# PARSE COMMANDLINE ARGUMENTS
opt = parser.parse_args()

