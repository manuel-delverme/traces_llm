import datetime
import inspect
import sys
from sys import platform as sys_pf

import matplotlib
import numpy as np

import experiment_buddy
import hyper

from dataset import DataSample

if sys_pf == 'darwin':
    matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def flatten_batch_and_sequence_dims(batch: DataSample, device):
    # Turn the timestep into another batch dimension
    # [batch, time, ...]  -> [batch * time, ...]

    images = batch.image_context.flatten(start_dim=0, end_dim=1)
    text_input = batch.next_token_logits.flatten(start_dim=0, end_dim=1)
    motor_context = batch.motor_context.flatten(start_dim=0, end_dim=1)
    labels = batch.labels.flatten(start_dim=0, end_dim=1)

    return DataSample(
        image_context=images.to(device),
        motor_context=motor_context.to(device),
        next_token_logits=text_input.to(device),
        labels=labels.to(device),
    )


# ---
# Demo for how to load image and stroke data for a character
# ---

# Plot the motor trajectory over an image
#
# Input
#  I [105 x 105 nump] grayscale image
#  drawings: [ns list] of strokes (numpy arrays) in motor space
#  lw : line width
def plot_motor_to_image(I, drawing, lw=2):
    drawing = [d[:, 0:2] for d in drawing]  # strip off the timing data (third column)
    drawing = [space_motor_to_img(d) for d in drawing]  # convert to image space
    plt.imshow(I, cmap='gray')
    ns = len(drawing)
    for sid in range(ns):  # for each stroke
        plot_traj(drawing[sid], get_color(sid), lw)
    plt.xticks([])
    plt.yticks([])


# Plot individual stroke
#
# Input
#  stk: [n x 2] individual stroke
#  color: stroke color
#  lw: line width
def plot_traj(stk, color, lw):
    n = stk.shape[0]
    if n > 1:
        plt.plot(stk[:, 0], stk[:, 1], color=color, linewidth=lw)
    else:
        plt.plot(stk[0, 0], stk[0, 1], color=color, linewidth=lw, marker='.')


# Color map for the stroke of index k
def get_color(k):
    scol = ['r', 'g', 'b', 'm', 'c']
    ncol = len(scol)
    if k < ncol:
        out = scol[k]
    else:
        out = scol[-1]
    return out


# convert to str and add leading zero to single digit numbers
def num2str(idx):
    if idx < 10:
        return '0' + str(idx)
    return str(idx)


# Load binary image for a character
#
# fn : filename
def load_img(fn):
    I = plt.imread(fn)
    I = np.array(I, dtype=bool)
    return I


# Load stroke data for a character from text file
#
# Input
#   fn : filename
#
# Output
#   motor : list of strokes (each is a [n x 3] numpy array)
#      first two columns are coordinates
#	   the last column is the timing data (in milliseconds)
def load_motor(fn):
    motor = []
    with open(fn, 'r') as fid:
        lines = fid.readlines()
    lines = [l.strip() for l in lines]
    for myline in lines:
        if myline == 'START':  # beginning of character
            stk = []
        elif myline == 'BREAK':  # break between strokes
            stk = np.array(stk)
            motor.append(stk)  # add to list of strokes
            stk = []
        else:
            arr = np.fromstring(myline, dtype=float, sep=',')
            stk.append(arr)
    return motor


#
# Map from motor space to image space (or vice versa)
#
# Input
#   pt: [n x 2] points (rows) in motor coordinates
#
# Output
#  new_pt: [n x 2] points (rows) in image coordinates
def space_motor_to_img(pt):
    pt[:, 1] = -pt[:, 1]
    return pt


def space_img_to_motor(pt):
    pt[:, 1] = -pt[:, 1]
    return


def calculate_wer(reference_tokens, hypothesis_tokens):
    """
    Calculate Word Error Rate (WER) between reference and hypothesis.

    Parameters:
    - reference (str): Ground truth sentence.
    - hypothesis (str): Predicted sentence.

    Returns:
    - wer (float): Word Error Rate.
    """

    # Initialize a table for dynamic programming
    dp = np.zeros((len(reference_tokens) + 1, len(hypothesis_tokens) + 1))

    dp[:, 0] = np.arange(len(reference_tokens) + 1)
    dp[0, :] = np.arange(len(hypothesis_tokens) + 1)

    for i in range(1, len(reference_tokens) + 1):
        for j in range(1, len(hypothesis_tokens) + 1):
            if reference_tokens[i - 1] == hypothesis_tokens[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = min(
                    dp[i - 1, j],  # Deletion
                    dp[i, j - 1],  # Insertion
                    dp[i - 1, j - 1]  # Substitution
                ) + 1

    wer = dp[len(reference_tokens)][len(hypothesis_tokens)] / len(reference_tokens)
    return wer


def buddy_setup():
    experiment_buddy.register_defaults(vars(hyper))
    import wandb
    wandb_kwargs = dict(
        monitor_gym=False, entity="delvermm", settings=wandb.Settings(start_method="thread"), save_code=True)
    # esh = ""
    # hostname = ""
    # sweep_config = ""
    proc_num = 4
    # hostname = "cc-beluga"
    # hostname = "cc-cedar"
    # hostname = "mila"
    hostname = "mila"
    # sweep_config = ""
    sweep_config = "sweep.yaml"
    # proc_num = -1
    # hostname = "aws://t4g.micro"
    if sys.gettrace() is not None and os.environ.get("BUDDY_DEBUG_DEPLOYMENT") is None:
        hostname = ""
        sweep_config = ""
    esh = "\n".join(l.strip() for l in """
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=64G
    #SBATCH --time=12:00:00
    #SBATCH --gres=gpu:32gb:1
        """.strip().split("\n")
                    ) + "\n"
    extra_modules = None
    if hostname == "mila":
        esh += "#SBATCH --partition=long\n"
        extra_modules = [
            "anaconda/3",
            "cuda/11.1",
            "pytorch/1.8.1"
        ]
    elif "cc" in hostname:
        esh += "#SBATCH --partition=cpubase_bycore_b4\n"
        esh += "#SBATCH --account=rrg-dprecup\n"
        # esh += "#SBATCH --account=rrg-bengioy-ad\n"
        extra_modules = [
            "anaconda/3",
            # "pytorch/1.7", # CC doesn't have pytorch, should be a package
            "cuda/11.1",
            "pytorch/1.8.1"
        ]
    else:
        esh = ""
    has_conda_env_param = inspect.signature(experiment_buddy.deploy).parameters.get("conda_env") is not None
    if has_conda_env_param:
        tb = experiment_buddy.deploy(
            hostname, wandb_kwargs=wandb_kwargs, extra_slurm_headers=esh, sweep_definition=sweep_config,
            proc_num=proc_num,
            extra_modules=extra_modules, conda_env="traces_llm"
        )
    else:
        tb = experiment_buddy.deploy(
            hostname, wandb_kwargs=wandb_kwargs, extra_slurm_headers=esh, sweep_definition=sweep_config,
            proc_num=proc_num,
            extra_modules=extra_modules
        )
    return tb


def timeit(f):
    def timed(*args, **kw):
        self = args[0]
        ts = datetime.datetime.now()
        result = f(*args, **kw)
        te = datetime.datetime.now()
        self.log(f"time/{f.__name__}", (te - ts).total_seconds(), batch_size=1)
        return result

    return timed
