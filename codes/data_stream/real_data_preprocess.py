import numpy as np


def real_data_preprocess(X_org):
    """ feature pre-process: [2021-11-23], updated on 2022-11-2
    The original #feature is 14, and the transformed #feature is reduced to 12.

    We pre-process the features according to Kamei's TSE2013.
    The implementation is from "2019 Local vs global models for jit-sdp - online clustering".
    But later in 2022-11, we found their codes with bugs and Liyan updated this func.
    Please see Liyan's report and NB-pg.95 for more details.

    Input & output arguments
        X_org: (n_sample, n_fea=14)
    return: preprocessed data_stream features
    """
    _, n_fea = X_org.shape
    assert n_fea == 14, "wrong dim of jit-sdp X_org"
    # manual setup carefully
    id_fix, id_ns, id_nd, id_nf, id_entropy = 0, 1, 2, 3, 4
    id_la, id_ld, id_lt, id_ndev, id_age = 5, 6, 7, 8, 9
    id_nuc, id_exp, id_rexp, id_sexp = 10, 11, 12, 13

    """Eliminate invalid entries from X_org. 
    The meanings of some features are annotated as below:
        * lt:   lines of code in a file before the change.
        * age:  the average time interval between the last and the current change.
        * rexp: recent developer experience
        * nf:   # modified files.
    Therefore, they should be non-negative in practice. 
    
    Also, negative values may induce technical errors. 
    For instance, in "log2", the potential log2(negative_value) will report two warnings:
        * RuntimeWarning: divide by zero encountered in log2
        * RuntimeWarning: invalid value encountered in log2 
    We may need to consider more features later on when dealing with more jit-sdp datasets.
    
    [2021-11-23] and updated on 2022-11-2.
    """
    use_data = np.logical_and(X_org[:, id_lt] >= 0,
                              X_org[:, id_age] >= 0,
                              X_org[:, id_rexp] >= 0)
    X_org = np.copy(X_org[use_data, :])
    X_trans = np.copy(X_org)

    """cumulative churn, Kamei's decChurn.r"""
    churn_np = (X_org[:, id_la] + X_org[:, id_ld]) / 2

    """feature pro-process"""
    # 1. deal with multi-collinearity
    # (1.1) LA = LA / LT; LD = LD / LT
    select_lt = X_trans[:, id_lt] >= 1  # avoid zero-denominator when lt==0
    X_trans[select_lt, id_la] = X_trans[select_lt, id_la] / X_trans[select_lt, id_lt]
    X_trans[select_lt, id_ld] = X_trans[select_lt, id_ld] / X_trans[select_lt, id_lt]

    # (1.2) LT = LT / NF; NUC = NUC / NF
    select_nf = X_trans[:, id_nf] >= 1  # avoid zero-denominator when nf=0
    X_trans[select_nf, id_lt] = X_trans[select_nf, id_lt] / X_trans[select_nf, id_nf]
    X_trans[select_nf, id_nuc] = X_trans[select_nf, id_nuc] / X_trans[select_nf, id_nf]

    # (1.3) entropy = entropy / NF   refer TSE'13 Kamei predUtils.r
    # [Kamei-TSE2013 code] if the num of files is less than 2, entropy is not normalized.
    select_nf = X_trans[:, id_nf] >= 2
    X_trans[select_nf, id_entropy] = X_trans[select_nf, id_entropy] / np.log2(X_trans[select_nf, id_nf])

    # (1.4) remove ND and REXP
    X_trans = X_trans[:, np.setdiff1d(range(n_fea), np.array((id_nd, id_rexp)))]

    # 2. logarithmic transformation
    n_fea_new = X_trans.shape[1]
    ids2_ = np.setdiff1d(range(n_fea_new), id_fix)  # Note that id_fix remains unchanged indeed.
    X_trans[:, ids2_] = X_trans[:, ids2_] + 1  # refer Kamei factorMain.r Line 38
    X_trans[:, ids2_] = np.log2(X_trans[:, ids2_])

    # 2022-11-7 churn should be aligned with X_trans
    # return X_trans, use_data, churn_np
    return np.hstack((X_trans, np.array([churn_np]).T)), use_data
