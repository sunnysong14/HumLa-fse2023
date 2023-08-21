import numpy as np
import numbers
import pandas as pd
import os
import shutil
import os.path

"""constant variables"""
ONE_DAY_HAS_TIMESTAMPS = 86400  # timestamps for one day
PSEUDO_LABEL_UNLABEL = -1  # pseudo label for unlabelled data_stream
MY_EPS = np.finfo("float").eps


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None, int or instance of RandomState
    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise, raise ValueError.
    Changwu Huang helped me 2022/4/18
    """
    # examples of usage:
    # rng = check_random_state(rnd_seed)
    # rng.choice(repo_id_use, nb_repo_request, replace=False)
    # or
    # rng.uniform(0, 1, 100)

    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState instance" % seed)


def cvt_day2timestamp(days: float) -> float:
    """
    Convert the days to the value of timestamps
    :param days: data_stream format in number, list, or tuple are all fine.
    :return: Unix timestamps
    """
    days_np = np.array(days)  # force to numpy data_stream
    day2timestamp_list = (days_np * ONE_DAY_HAS_TIMESTAMPS).tolist()
    return day2timestamp_list


def cvt_timestamp2day(timestamps):
    timestamps_np = np.array(timestamps)
    days_list = (timestamps_np / ONE_DAY_HAS_TIMESTAMPS).tolist()
    return days_list


def load_para_csv():
    """
    Load the tuned parameters that were done in preliminary experiments.
    One can make their own parameter settings if they want.
    """
    csv_dir = "data_stream/"
    para_csv = pd.read_csv(csv_dir + 'para_bst_30sd.csv')
    return para_csv


def delete_invalid_folders(post_fix="careful"):
    """
    delete invalid (sub-)folders as defective codes may generate cached result folders that become invalid and thus
    should be eliminated.
    TC and Liyan latest updated on 2022/12
    """
    # PLS be careful to use this function.
    root_dir = "../results/rslt.save"
    pre_fix = "cache."  # for safety reason to fix this string
    invalid_folder_name = pre_fix + post_fix
    for parent, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if dirname == invalid_folder_name:
                del_folder_name = parent + os.sep + dirname
                shutil.rmtree(del_folder_name)
                print(del_folder_name + " has been deleted.")


if __name__ == '__main__':
    times = [86401, 186400, 286400]
    print(cvt_timestamp2day(times))

