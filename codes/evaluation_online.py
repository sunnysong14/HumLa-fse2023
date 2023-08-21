import numpy as np
import warnings

# silence the warning
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')


def eval_clf_online(result_np, theta_pf=0.99):
    """for pp-report
    @result_np: (time, y_true, y_pred), created in xxx.jit_sdp_1call()
    @theta_pf: the para theta_imb to evaluate the online pf
    Liyan on 2021-10-12
    """
    # print(info_str) >> the index is 0: time, 1: y_true, 2: y_pred
    actual_labels = result_np[:, 1]
    predict_labels = result_np[:, 2]

    recall0_tt, recall1_tt, gmean_tt = compute_online_PF(actual_labels, predict_labels, theta_pf)
    gmean_tt = gmean_tt.reshape(len(gmean_tt))
    recall0_tt = recall0_tt.reshape(len(recall0_tt))
    recall1_tt = recall1_tt.reshape(len(recall1_tt))
    # pp-report, (3, #steps)
    pf_metrics_tt = np.vstack((gmean_tt, recall0_tt, recall1_tt))

    # ave across time steps
    gmean_ave = np.nanmean(gmean_tt, 0)
    recall0_ave = np.nanmean(recall0_tt, 0)
    recall1_ave = np.nanmean(recall1_tt, 0)
    # pp-report, 1*3
    pf_metrics_ave = (gmean_ave, recall0_ave, recall1_ave)

    metric_names = np.array(("gmean", "recall0", "recall1"))
    return pf_metrics_tt, pf_metrics_ave, metric_names


def Gmean_compute(recall):
    Gmean = 1
    for r in recall:
        Gmean = Gmean * r
    Gmean = pow(Gmean, 1/len(recall))
    return Gmean


def avg_acc_compute(recall):
    avg_acc = np.mean(recall)
    return avg_acc


def f1_compute(tr, prec, pos_class=1):
    assert pos_class == 1 or pos_class == 0, "current version on 20221201 only works for binary class"
    f1_score = 2 * tr * prec / (tr + prec)
    return f1_score[pos_class]


def mcc_compute(tr, fr, pos_class=1):
    """
    The implementation is based on https://blog.csdn.net/Winnycatty/article/details/82972902
    The undefined MCC that a whole row or column of the confusion matrix M is zero is treated the left column of page 5
    of the paper: Davide Chicco and Giuseppe Jurman. "The advantages of the matthews correlation coefficient (mcc) over
        f1 score and accuracy in binary classification evaluation". BMC Genomics, 21, 01, 2020
    The confusion matrix M is
        M = (tp fn
             fp tn)

    TC on 2022/11/25, TC update 2022/11/28
    Liyan update 2022/12/1
    """
    fenzi = tr[0] * tr[1] - fr[0] * fr[1]
    tp = tr[pos_class]  # defined positive_class is positive
    tn = tr[1 - pos_class]
    fn = fr[pos_class]
    fp = fr[1 - pos_class]
    fenmu = pow((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn), 0.5)
    if fenmu == 0:
        if (tp or tn) and fn == 0 and fp == 0:  # M has only 1 non-0 entry & all are correctly predicted
            mcc = 1
        elif (fp or fn) and tp == 0 and tn == 0:  # M has only 1 non-0 entry & all are incorrectly predicted
            mcc = -1
        else:  # a row or a column of M are zero
            mcc = 0
    else:
        mcc = fenzi/fenmu
    return mcc


def pf_epoch(S, N, P, theta, t, y_t, p_t, pos_class=1):
    """ Reference:
    Gama, Joao, Raquel Sebastiao, and Pedro Pereira Rodrigues.
    "On evaluating stream learning algorithms." Machine learning 90.3 (2013): 317-346.

    Shuxian update on 2022/12/1
    """
    if t == 0:
        c = int(y_t)  # class 0 or 1
        S[t, c] = (y_t == p_t)
        N[t, c] = 1
        P[t, c] = 1
    else:
        S[t, :] = S[t-1, :]
        N[t, :] = N[t-1, :]
        P[t, :] = P[t-1, :]
        c = int(y_t)  # class 0 or 1
        S[t, c] = (y_t == p_t) + theta * (S[t-1, c])
        N[t, c] = 1 + theta * N[t-1, c]
        p = int(p_t)  # the number of predicted positive data
        P[t, p] = 1 + theta * P[t-1, p]

    recall = S[t, :] / N[t, :]

    assert pos_class == 1 or pos_class == 0, "current version on 20221201 only works for binary class"
    tr = recall  # positive class is 1, then tpr = tr[1], tnr = tr[0]
    fr = 1 - tr  # positive class is 1, then fnr = fr[1], fpr = fr[0]
    prec = tr / (tr + np.flip(fr))
    prec = prec[pos_class]
    f1_score = f1_compute(tr, prec)
    mcc = mcc_compute(tr, fr)
    gmean = Gmean_compute(recall)
    ave_acc = avg_acc_compute(recall)
    return recall, gmean, mcc, prec, f1_score, ave_acc


def compute_online_PF(y_tru, y_pre, theta_eval=0.99):
    """
    para theta_eval: used in the online PF evaluation, theta_eval=0.99 by default
    reference: 2013_[JML, #, Leandro based] On evaluate stream learning algorithm

    Liyan and Shuxian created on 2021/9, last updated on 2022/11/25
    """
    S = np.zeros([len(y_tru), 2])
    N = np.zeros([len(y_tru), 2])
    P = np.zeros([len(y_tru), 2])
    recalls_tt = np.empty([len(y_tru), 2])
    Gmean_tt = np.empty([len(y_tru), ])
    ave_acc_tt = np.empty([len(y_tru), ])
    prec_tt = np.empty([len(y_tru), ])
    f1_tt = np.empty([len(y_tru), ])
    mcc_tt = np.empty([len(y_tru), ])
    # compute at each test step
    for t in range(len(y_tru)):
        y_t = y_tru[t]
        p_t = y_pre[t]
        recalls_tt[t, :], Gmean_tt[t], mcc_tt[t], prec_tt[t], f1_tt[t], ave_acc_tt[t] \
            = pf_epoch(S, N, P, theta_eval, t, y_t, p_t)
        recall0_tt = recalls_tt[:, 0]
        recall1_tt = recalls_tt[:, 1]
    # assign pfs
    pfs_dct = dict()
    pfs_dct["gmean_tt"] = Gmean_tt
    pfs_dct["recall1_tt"], pfs_dct["recall0_tt"] = recall1_tt, recall0_tt
    pfs_dct["mcc_tt"] = mcc_tt
    pfs_dct["precision_tt"] = prec_tt
    pfs_dct["f1_score_tt"] = f1_tt
    pfs_dct["ave_acc_tt"] = ave_acc_tt
    return pfs_dct


if __name__ == '__main__':
    theta = 0.99
    y = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
    p = [0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    pfs_dct = compute_online_PF(y, p, theta)
    recall0 = pfs_dct["recall0_tt"]
    recall1 = pfs_dct["recall1_tt"]
    Gmean = pfs_dct["gmean_tt"]
    mcc = pfs_dct["mcc_tt"]
    precision = pfs_dct["precision_tt"]
    f1_score = pfs_dct["f1_score_tt"]
    avg_acc = pfs_dct["ave_acc_tt"]
    # print
    print('Gmean: ', np.nanmean(Gmean))
    print('mcc: ', np.nanmean(mcc))
    print('recall0: ', np.nanmean(recall0))
    print('recall1: ', np.nanmean(recall1))
    print('precision: ', np.nanmean(precision))
    print('f1_score: ', np.nanmean(f1_score))
    print('avg_acc: ', np.nanmean(avg_acc))
