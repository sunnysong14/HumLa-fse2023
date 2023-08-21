from skmultiflow.data.file_stream import FileStream
from codes.utility import cvt_day2timestamp, load_para_csv
import numpy as np

""" JIT-SDP data_stream:
As we mainly use Hoeffding tree for classification, there is no need to normalize data_stream features.
However, if later on we decide to adopt other ML methods such as distance-based clustering,
we should be careful that feature normalization may be required.
Liyan Song 
Latest updated on 2022/8/9
"""


class class_data_ind_org:
    # Inputs contains 14 features that will then transform to 12 fea-s acc Kamei-TSE2013.
    # The codes will auto reset the index info for the 13 transformed fea-s.
    def __init__(self):
        self.nb_fea = 14
        self.nb_inputs = self.nb_fea + 2  # 14-fea + 1 commit time + 1 VL
        self.id_time = 0
        self.id_vl = -1
        self.id_X_np = np.s_[:, self.id_time + 1:self.id_vl]


class class_data_ind_reset:
    # Manual rest data_stream index after fea conversion. 2022/8/8
    # After fea conversion: col=3+12 ~ (time, #fea12, #1churn, vl, yy)
    def __init__(self, id_time, id_y, id_vl, id_churn, id_X_np: np, n_fea=12):
        self.id_time = id_time
        self.id_vl = id_vl
        self.id_y = id_y
        self.id_X_np = id_X_np
        self.id_churn = id_churn
        self.n_fea = n_fea


def data_id_2name(project_id):
    if project_id == 0:
        project_name = "brackets"
    elif project_id == 1:
        project_name = "broadleaf"
    elif project_id == 2:
        project_name = "camel"
    elif project_id == 3:
        project_name = "fabric"
    elif project_id == 4:
        project_name = "jgroups"
    elif project_id == 5:
        project_name = "nova"
    elif project_id == 6:
        project_name = "tomcat"
    elif project_id == 7:
        project_name = "corefx"
    elif project_id == 8:
        project_name = "django"
    elif project_id == 9:
        project_name = "rails"
    elif project_id == 10:
        project_name = "rust"
    elif project_id == 11:
        project_name = "tensorflow"
    elif project_id == 12:
        project_name = "vscode"
    elif project_id == 13:
        project_name = "wp-calypso"
    else:
        raise Exception("undefined data_stream id.")
    return project_name


def lookup_best_para(data_name, wait_days, clf_name, para_csv=None):
    """
    Look up the best parameter settings of some datasets in the preliminary experiments that has been done for ijcnn'22.
    If para_best are not found, an error arises.
    Anonymous 2022/10/31
    """
    # input arguments
    if para_csv is None:
        para_csv = load_para_csv()
    data_name = data_name.lower()
    clf_name = clf_name.lower()

    # core to look up
    mask = (para_csv.iloc[:, 0] == data_name) & (para_csv.iloc[:, 1] == wait_days) & (para_csv.iloc[:, 2] == clf_name)
    data_extracted = para_csv.loc[mask, :]
    if data_extracted.shape[0]:
        n_tree_bst = data_extracted.iloc[-1, 3]
        theta_imb_bst = data_extracted.iloc[-1, 4]
        theta_cl_bst = data_extracted.iloc[-1, 5]
    else:
        raise Exception("Existing para_best.csv does NOT contain this tuned para_best.")
    return n_tree_bst, theta_imb_bst, theta_cl_bst


def set_test_stream(project_name):
    """ load_test_stream
    Load the test data_stream stream prepared in MATLAB previously.
    Note that data_stream XX should have been sorted acc commit timestamps in ascending order already.
    @project_name: str, project name
    return: numpy, format - ((ts,XX,vl), y)

    Liyan Song songly@sustech.edu.cn
    created on 2021-7-14
    """
    dir_load_data = "../dataset/data.inuse/"
    # data_stream: (ts, XX; y; vl)
    data_test_stream = FileStream(dir_load_data + project_name + "_vld_st.csv", target_idx=-2)
    # see skmultiflow.data_stream.file_stream.FileStream for how to use the data_test_stream
    return data_test_stream


def set_train_stream(
        prev_test_time, curr_test_time, new_data, data_ind: class_data_ind_reset, data_buffer=None, wait_days=30):
    """ set training stream for jit-sdp
    Inputs:
        new_data: numpy, (n_sample, n_col) where n_col~(time, 12-fea, vl, yy), see data_ind
        data_ind: class_data_ind_reset
    Log:
        2021-11-1   retains only the "delay-noisy" case for JTI-SDP.
        2022-7-28   insert the class "data_ind"
    """

    # get data_stream index
    id_time, id_vl, id_y, id_churn, id_X = \
        data_ind.id_time, data_ind.id_vl, data_ind.id_y, data_ind.id_churn, data_ind.id_X_np
    if new_data.ndim == 1:  # debug
        new_data = new_data.reshape((1, -1))

    """store new_data into data_buffer~(time, 12-fea, vl, y)"""
    for dd in range(new_data.shape[0]):
        data_1new = new_data[dd, :].reshape((1, -1))
        # [vip] overwrite clean data_stream's VL to np.inf
        if data_1new[0, id_y] == 0:
            data_1new[0, id_vl] = np.inf
        # set data_buffer, (ts, XX, vl)
        if data_buffer.shape[0] == 0:  # init
            data_buffer = data_1new
        else:
            data_buffer = np.vstack((data_buffer, data_1new))

    """create / update the training sets.
    Consider VL and label noise: if there are labeled training XX becomes available 
    between the last test step and current test step, set the defect and the clean data_stream sets, 
    and maintain the data_buffer carefully.
    """
    # 1) set train_data_defect and update data_buffer
    is_defect = curr_test_time > (data_buffer[:, id_time] + cvt_day2timestamp(data_buffer[:, id_vl]))
    train_data_defect = data_buffer[is_defect, :]
    # update data_buffer: pop out defect-inducing data_stream
    data_buffer = data_buffer[~is_defect, :]  # (time, 12-fea, churn vl, y)

    # 2) set train_data_clean and update data_buffer
    wait_days_clean_upp = curr_test_time > data_buffer[:, id_time] + cvt_day2timestamp(wait_days)
    wait_days_clean_low = prev_test_time <= data_buffer[:, id_time] + cvt_day2timestamp(wait_days)
    wait_days_clean = wait_days_clean_low & wait_days_clean_upp
    train_data_clean = data_buffer[wait_days_clean, :]  # possible label noise

    # VIP update data_buffer: pop out the 'real' clean data_stream
    actual_clean = data_buffer[:, id_y] == 0
    # actual_clean = np.isinf(data_buffer[:, ID_vl])  # NOTE clean data_stream's VL should have been assigned to np.inf
    wait_actual_clean = wait_days_clean & actual_clean
    data_buffer = data_buffer[~wait_actual_clean, :]  # (ts, 12-fea, churn vl, y)

    # 3) set train_data_unlabeled, no need to update data_buffer
    idx_upp_time_unlabeled = data_buffer[:, id_time] < curr_test_time
    lowest_time = max(prev_test_time, curr_test_time - cvt_day2timestamp(wait_days))
    idx_low_time_unlabeled = data_buffer[:, id_time] >= lowest_time
    train_data_unlabeled = data_buffer[idx_upp_time_unlabeled & idx_low_time_unlabeled]

    return data_buffer, train_data_defect, train_data_clean, train_data_unlabeled


if __name__ == "__main__":
    data_stream = set_test_stream("bracket")
