import sys
import numpy as np
from sklearn.utils import check_array
from copy import copy
from DenStream.MicroCluster import MicroCluster
from math import ceil
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt


"""constant variables"""
MY_EPS = np.finfo("float").eps


class DenStream:
    """ 2021-11-19
    Liyan Song downloaded this from: https://github.com/waylongo/denstream
    Liyan also needed to incorporate our proposed CL algo with the DenStream framework.
    """

    def __init__(self, theta_cl=0.8, lambd=1, eps=1, beta=2, mu=2):
        """
        DenStream - Density-Based Clustering over an Evolving Data Stream with
        Noise.

        Parameters
        ----------
        lambd: float, optional
            The forgetting factor. The higher the value of lambda, the lower
            importance of the historical X_org compared to more recent X_org.
        eps : float, optional
            The maximum distance between two XX for them to be considered
            as in the same neighborhood.

        Attributes
        ----------
        labels_ : array, shape = [n_samples]
            Cluster yy for each point in the dataset given to fit().
            Noisy XX are given the label -1.

        Notes
        -----


        References
        ----------
        Feng Cao, Martin Estert, Weining Qian, and Aoying Zhou. Density-Based
        Clustering over an Evolving Data Stream with Noise.
        """
        self.lambd = lambd
        self.eps = eps
        self.beta = beta
        self.mu = mu
        self.t = 0  # 2021-11-24 TODO # of used X_org?
        self.p_micro_clusters = []
        self.o_micro_clusters = []
        if lambd > 0:
            # print("We should follow the rule: beta*mu should > 1")
            self.tp = ceil((1 / lambd) * np.log((beta * mu) / (beta * mu - 1)))
        else:
            self.tp = sys.maxsize

        """Liyan: entire X_org info (all micro-clusters)"""
        self.nb_all_clean = 0
        self.nb_all_defect = 0
        self.nb_all_unlabelled = 0
        self.def_data_repo = self.DefDataRepo()
        self.theta_cl = theta_cl  # CL encoded into oob

    def partial_fit(self, X, y, sample_weight=None):
        """
        Liyan Song
        2021-12-6   the label value of unlabelled X_org is auto tackled here, no worries.

        ================================================
        Online learning.

        XX : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of training X_org
        y : Ignored
        sample_weight : array-like, shape (n_samples,), optional
            Weights applied to individual XX.
            If not provided, uniform weights are assumed.

        Returns
        -------
        self : returns an instance of self.
        """

        X = check_array(X, dtype=np.float64, order="C")
        n_samples, _ = X.shape
        sample_weight = self._validate_sample_weight(sample_weight, n_samples)
        # if not hasattr(self, "potential_micro_clusters"):
        # if n_features != :
        # raise ValueError("Number of features %d does not match previous "
        # "X_org %d." % (n_features, self.coef_.shape[-1]))

        # core partial fit
        for one_X, one_y, weight in zip(X, y, sample_weight):
            self._partial_fit(one_X, one_y, weight)

        # update nb_data_cluster
        nb_clean = np.count_nonzero(y == 0)
        nb_defect = np.count_nonzero(y == 1)
        nb_unlabelled = len(y) - nb_clean - nb_defect
        self.nb_all_clean += nb_clean
        self.nb_all_defect += nb_defect
        self.nb_all_unlabelled += nb_unlabelled
        return self

    def predict(self, X):
        """ Predict each of XX into a micro-cluster.
        :para XX: shape (n_sample, N_fea)
        :return: numpy (n_sample, _)
        In DenStream, it happens that a X_org does not belong to any micro-cluster.
        We simply find its closest micro-cluster as its pseudo-cluster label.
        This is normally OK for the DenStream framework,
        but it may cause error for update_cluster_info()
        ==========================
        2021-11-24: Liyan forms it based on fit_predict()
        """
        y = []
        for sample in X:
            index, _ = self._get_nearest_micro_cluster(sample, self.p_micro_clusters)
            y.append(index)
        return np.asarray(y)

    def fit_predict(self, X, y=None, sample_weight=None):
        """
        Liyan 2021-11-19: I think this method can be replaced by partial_fit() and predict()
        ===============================
        Lorem ipsum dolor sit amet

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of training X_org

        y : Ignored

        sample_weight : array-like, shape (n_samples,), optional
            Weights applied to individual XX.
            If not provided, uniform weights are assumed.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Cluster yy
        """

        X = check_array(X, dtype=np.float64, order="C")

        n_samples, _ = X.shape

        sample_weight = self._validate_sample_weight(sample_weight, n_samples)

        # if not hasattr(self, "potential_micro_clusters"):

        # if n_features != :
        # raise ValueError("Number of features %d does not match previous "
        # "X_org %d." % (n_features, self.coef_.shape[-1]))

        for sample, y_one, weight in zip(X, y, sample_weight):
            self._partial_fit(sample, y_one, weight)

        p_micro_cluster_centers = np.array([p_micro_cluster.center() for
                                            p_micro_cluster in
                                            self.p_micro_clusters])
        p_micro_cluster_weights = [p_micro_cluster.weight() for p_micro_cluster in
                                   self.p_micro_clusters]
        dbscan = DBSCAN(eps=5, algorithm='brute')
        dbscan.fit(p_micro_cluster_centers,
                   sample_weight=p_micro_cluster_weights)  # TODO Shuxian: cluster for centers

        y = []
        for sample in X:
            index, _ = self._get_nearest_micro_cluster(sample, self.p_micro_clusters)
            y.append(index)  # 2021-11-22 Liyan changed

        return y

    def _get_nearest_micro_cluster(self, sample, micro_clusters):
        smallest_distance = sys.float_info.max
        nearest_micro_cluster = None
        nearest_micro_cluster_index = -1
        for i, micro_cluster in enumerate(micro_clusters):
            current_distance = np.linalg.norm(micro_cluster.center() - sample)
            if current_distance < smallest_distance:
                smallest_distance = current_distance
                nearest_micro_cluster = micro_cluster
                nearest_micro_cluster_index = i
        return nearest_micro_cluster_index, nearest_micro_cluster

    def _try_merge(self, sample, y, weight, micro_cluster):
        if micro_cluster is not None:
            micro_cluster_copy = copy(micro_cluster)
            micro_cluster_copy.insert_sample(sample, y, weight)
            if micro_cluster_copy.radius() <= self.eps:
                micro_cluster.insert_sample(sample, y, weight)
                return True
        return False

    def _merging(self, sample, y, weight):
        # (1) Try to merge the sample with its nearest p_micro_cluster
        _, nearest_p_micro_cluster = \
            self._get_nearest_micro_cluster(sample, self.p_micro_clusters)
        success = self._try_merge(sample, y, weight, nearest_p_micro_cluster)
        if not success:
            # (2) Try to merge the sample into its nearest o_micro_cluster
            index, nearest_o_micro_cluster = \
                self._get_nearest_micro_cluster(sample, self.o_micro_clusters)
            success = self._try_merge(sample, y, weight, nearest_o_micro_cluster)
            if success:
                if nearest_o_micro_cluster.weight() > self.beta * self.mu:
                    del self.o_micro_clusters[index]
                    self.p_micro_clusters.append(nearest_o_micro_cluster)
            else:
                # Create new o_micro_cluster
                micro_cluster = MicroCluster(self.lambd, self.t)
                micro_cluster.insert_sample(sample, y, weight)
                self.o_micro_clusters.append(micro_cluster)

    def _decay_function(self, t):
        return 2 ** ((-self.lambd) * (t))

    def _partial_fit(self, sample, y, weight):
        """Note that sample size = 1
        ==============
        2021-11-24 Liyan added some comments.
        """

        # core merging
        self._merging(sample, y, weight)

        # check every Tp time periods, see DenStream pp Eq.(4.1)
        if self.t % self.tp == 0:
            # for p_micro_clusters
            self.p_micro_clusters = [p_micro_cluster for p_micro_cluster in self.p_micro_clusters
                                     if p_micro_cluster.weight() >= self.beta * self.mu]
            # for o_micro_clusters, see pp Eq.(4.2)
            Xis = [((self._decay_function(self.t - o_micro_cluster.creation_time + self.tp) - 1) /
                    (self._decay_function(self.tp) - 1)) for o_micro_cluster in self.o_micro_clusters]
            self.o_micro_clusters = [o_micro_cluster for Xi, o_micro_cluster in
                                     zip(Xis, self.o_micro_clusters) if
                                     o_micro_cluster.weight() >= Xi]
        self.t += 1  # TODO ?

    def _validate_sample_weight(self, sample_weight, n_samples):
        """Set the sample weight array."""
        if sample_weight is None:
            # uniform sample weights
            sample_weight = np.ones(n_samples, dtype=np.float64, order='C')
        else:
            # user-provided array
            sample_weight = np.asarray(sample_weight, dtype=np.float64,
                                       order="C")
        if sample_weight.shape[0] != n_samples:
            raise ValueError("Shapes of XX and sample_weight do not match.")
        return sample_weight

    def micro_clusters_bias(self):
        """ Core part of my algorithm. Liyan added this.
        Compute the micro-mc_cluster bias towards class label 0 and 1, respectively.
        :return bias_2_c0: shape (# micro-cluster, _)
        :return bias_2_c1: shape (# micro-cluster, _)

        2021-11-24 copied from CluStream
        2022/4/11  for ooc extension
        """
        # 2021-11-8 we choose option-1
        opt_ = 0
        if opt_ == 0:  # only labelled X_org
            nb_all_data = self.nb_all_clean + self.nb_all_defect
        elif opt_ == 1:  # all X_org
            nb_all_data = self.nb_all_clean + self.nb_all_defect + self.nb_all_unlabelled
        prob_c0 = self.nb_all_clean / nb_all_data
        prob_c1 = self.nb_all_defect / nb_all_data

        """1. primary micro-cluster bias"""
        bias_2_c0 = np.empty(len(self.p_micro_clusters))
        bias_2_c1 = np.copy(bias_2_c0)
        for cc, mc_cluster in enumerate(self.p_micro_clusters):
            if opt_ == 0:
                nb_mc_data = mc_cluster.nb_mc_clean + mc_cluster.nb_mc_defect
            elif opt_ == 1:
                nb_mc_data = mc_cluster.nb_mc_clean + mc_cluster.nb_mc_defect + mc_cluster.nb_mc_unlabelled
            prob_mc0 = mc_cluster.nb_mc_clean / nb_mc_data
            prob_mc1 = mc_cluster.nb_mc_defect / nb_mc_data
            # [report 2021/11] implement when "nb_mc_labeled=0"
            bias_2_c0[cc] = mc_cluster.class_bias(prob_mc0, prob_c0)
            bias_2_c1[cc] = mc_cluster.class_bias(prob_mc1, prob_c1)

        """2. rescale micro-cluster bias:  [2021-11-10] Shuxian
        for class-0 and class-1, normalise the bias values across micro-clusters"""
        if np.sum(bias_2_c0) != 0:
            bias_2_c0 = np.divide(bias_2_c0, np.sum(bias_2_c0))
        if np.sum(bias_2_c1) != 0:
            bias_2_c1 = np.divide(bias_2_c1, np.sum(bias_2_c1))

        return bias_2_c0, bias_2_c1

    def weights_to_points(self, data_points):
        """ Compute weights of data_points to each micro-cluster
        :para: data_points shape (n_sample, N_fea)
        2021-11-24 copy from Clustream for DenStream and revises
        """

        if data_points.ndim == 1:  # debug
            data_points = data_points.reshape(1, len(data_points))
        nb_data = data_points.shape[0]

        """1. derive similarity and concept-drift betw each X_org and micro-cluster"""
        mc_cluster_sim = np.empty((nb_data, len(self.p_micro_clusters)))  # init
        mc_cluster_time = np.copy(mc_cluster_sim)  # init
        for dd, data_point in enumerate(data_points):
            for cc, mc_cluster in enumerate(self.p_micro_clusters):
                mc_cluster_sim[dd, cc], mc_cluster_time[dd, cc] = mc_cluster.weight_to_1point(data_point)

        """2. primary micro-cluster weights across all X_org points"""
        mc_cluster_weights = np.multiply(mc_cluster_sim, mc_cluster_time)
        mc_cluster_weights[mc_cluster_weights < MY_EPS] = 0

        """3. micro-cluster weight pruning [2021-11-8]
        each X_org has its own threshold, determine heuristically"""
        for dd in range(nb_data):
            weights_4_data = mc_cluster_weights[dd, :]
            ave_, std_ = np.mean(weights_4_data), np.std(weights_4_data)
            threshold_prune = max(ave_ - 1 * std_, MY_EPS)  # heuristic, 68%-percentile
            mc_cluster_weights[dd, np.where(weights_4_data <= threshold_prune)[0]] = 0

        return mc_cluster_weights

    class DefDataRepo:
        """
        The X_org repository for the latest defect-inducing X_org.
        This is used in the refinement step of our core CL algorithm.
        ==================
        2021-11-15 Liyan Song created for CluStream
        2021-11-24 copied from CluStream
        """

        def __init__(self, time=None, X_norm=None, y_obv=None, max_size=8):
            self.max_size = max_size  # hyper-para
            self.time = time
            self.X_norm = X_norm
            self.y_obv = y_obv

        def check_data_repo(self):
            label_values = [1]
            assert all(np.unique(self.y_obv) == label_values), "label values should be (0, 1)."

        def is_repo_empty(self):
            if self.y_obv is None:
                return True
            elif len(self.y_obv) == 0:
                return True
            else:
                return False

        def get_data_num(self):
            if self.y_obv is not None:
                nb_data = len(self.y_obv)
            else:
                nb_data = 0
            return nb_data

        def insert_def_data(self, new_X_norm, new_y_obv, new_y_time):
            """This method can choose and insert defect X_org into the repository automatically,
            and maintains the predefined repository size.
            2021-11-15
            """
            new_y_obv = np.atleast_1d(new_y_obv)
            new_y_time = np.atleast_1d(new_y_time)

            # find new defect X_org
            idx_def_ = np.where(new_y_obv == 1)[0]
            new_X_norm = new_X_norm[idx_def_, :]
            new_y_obv = new_y_obv[idx_def_]
            new_time = new_y_time[idx_def_]

            # add defect X_org into the repo
            if self.time is None:  # pre-train
                self.time, self.X_norm, self.y_obv = new_time, new_X_norm, new_y_obv
            else:
                self.time = np.concatenate((self.time, new_time))
                self.X_norm = np.vstack((self.X_norm, new_X_norm))
                self.y_obv = np.concatenate((self.y_obv, new_y_obv))

            # only retain latest defect X_org IF exceed max_size
            if self.get_data_num() > self.max_size:
                idx_retain_ = np.argpartition(self.time, -self.max_size)[-self.max_size:]  # top-k
                self.time = self.time[idx_retain_]
                self.X_norm = self.X_norm[idx_retain_, :]
                self.y_obv = self.y_obv[idx_retain_]

        def pop_def_data(self, nb_pop):
            if nb_pop > self.get_data_num():
                print("\tAttention: # required pop def X_org > the repository size. Pop out all we have.")
            nb_pop_fnl = min((nb_pop, self.get_data_num()))
            idx_pop_fnl = np.argpartition(self.time, -nb_pop_fnl)[-nb_pop_fnl:]  # top-k
            pop_X_norm = self.X_norm[idx_pop_fnl, :]
            pop_time = self.time[idx_pop_fnl]
            pop_y_obv = self.y_obv[idx_pop_fnl]

            self.check_data_repo()
            return pop_X_norm, pop_y_obv, pop_time, nb_pop_fnl

    def check_data_dim(self, X_norm, y_obv):
        if X_norm.ndim == 2:
            assert X_norm.shape[0] == y_obv.shape[0], "the #X_org NOT match #label"
        elif X_norm.ndim == 1:
            assert np.atleast_1d(y_obv).shape[0] == 0, "the #X_org NOT match #label"

    def compute_CLs(self, X_norm, y_obv, nb_min_c1_refn=5):
        """ the main CL framework:
        Compute CLs of observed yy of training X_org.
        Please refer to the Slide/Report for the implementation details.

        :param X_norm: numpy of (nb_data_, nb_feature)
        :param y_obv: numpy of (nb_data_, )
        :param nb_min_c1_refn: minimum number of defect X_org required for the CL refinement
                               2021-11-29 decide not to take this module
        :returns CL_y_obv, CLs_c1_refine

        Liyan Song: songly@sustech.edu.cn
        2021-11-24 copied from CluStream
        """
        self.check_data_dim(X_norm, y_obv)

        if len(self.p_micro_clusters) < 0:  # no exist_cluster
            raise Exception("Non micro-cluster is constructed, so cannot compute CLs now.")

        def primary_CLs(y_obv, bias_c0_step1_2, bias_c1_step1_2, weight_data_2_cluster):
            """[2021-11-15] step1-4 sub-function within self.compute_label_CLs()
            """
            my_tolerance = 0.00001  # threshold if a X_org point belongs to any p-cluster
            nb_data = len(y_obv)
            CLs_c0_primary = np.empty(nb_data)  # init
            CLs_c1_primary = np.empty(nb_data)  # init
            use_cluster = np.empty(nb_data)  # init, report info

            """implementation of step 1-4.
            [2021-11-29] revise to handle case-0 that "no micro-cluster wants this point".
            Previously, we treat the case "sum_c0 + sum_c1=0" as a "bug", treat it as a special case, 
            and handle it with codes. Now we treat this special issue more reasonable.
            However, surprisingly experimental evaluate showed that the newly proposed strategy can often 
            cause side effect on the predictive PF in JIT-SDP. 
            Thus, I am a bit hesitated to use this mechanism, though it sounds very reasonable"""
            for dd, y_1obv in enumerate(y_obv):
                clu_weights_data = weight_data_2_cluster[dd, :]
                if np.sum(clu_weights_data) >= my_tolerance:
                    use_cluster[dd] = True
                    sum_c0 = np.sum(np.multiply(bias_c0_step1_2, clu_weights_data))
                    sum_c1 = np.sum(np.multiply(bias_c1_step1_2, clu_weights_data))
                    sum_c0c1 = sum_c0 + sum_c1
                    """[2021-11-29] we suppose that micro-clusters' biases should not be all 0,
                    so sum_c0c1 should not be 0."""
                    assert sum_c0 >= 0 and sum_c1 >= 0 and sum_c0c1 > 0, (sum_c0, sum_c1)
                    # 1-sum normalization
                    if sum_c0c1 > my_tolerance:
                        CLs_c0_primary[dd] = sum_c0 / sum_c0c1
                        CLs_c1_primary[dd] = sum_c1 / sum_c0c1
                    else:
                        # todo rearrange this handle with Shuxian, 2021-12-2
                        # raise Exception("TODO [2021-12-2] we should not arrive here.")
                        # tmp code below
                        use_cluster[dd] = False
                        if y_1obv == 1:
                            CLs_c0_primary[dd] = 0
                            CLs_c1_primary[dd] = 1
                        elif y_1obv == 0:
                            n_all_data = self.nb_all_clean + self.nb_all_defect
                            CLs_c0_primary[dd] = self.nb_all_clean / n_all_data
                            CLs_c1_primary[dd] = self.nb_all_defect / n_all_data
                else:
                    """[2021-11-25] No micro-cluster wants this X_org.
                    The heuristic strategy:
                    - if y_obv == 1: cl_c1_primary = 1, cl_c0_primary = 0
                    - if y_obv == 0: cl_c1_primary = 0, cl_c0_primary = Pr(class=0)"""
                    use_cluster[dd] = False
                    if y_1obv == 1:
                        CLs_c0_primary[dd] = 0
                        CLs_c1_primary[dd] = 1
                    elif y_1obv == 0:
                        n_all_data = self.nb_all_clean + self.nb_all_defect
                        CLs_c0_primary[dd] = self.nb_all_clean / n_all_data
                        CLs_c1_primary[dd] = self.nb_all_defect / n_all_data
                    else:
                        raise Exception("y_obv=%d can only be 0 or 1." % y_obv)

            return CLs_c0_primary, CLs_c1_primary, use_cluster

        """step 1-1. online cluster -- conducted outside in CluStream()"""
        """step 1-2. micro-cluster representativeness for class 0 and class 1"""
        bias_c0, bias_c1 = self.micro_clusters_bias()
        # [record 2021/11] implement time-decayed class size of each micro-cluster, see Report

        """step 1-3. towards micro-clusters' weight:
        micro-cluster pruning strategy is employed here"""
        weight_data_2_clusters = self.weights_to_points(X_norm)
        # [p_cluster.weight() for p_cluster in self.p_micro_clusters]

        """step 1-4. class biases (towards class-0 and class-1) across X_org points.
        The issue that a X_org does not belong to any micro-cluster is handled here"""
        CLs_c0, CLs_c1, use_cluster = primary_CLs(y_obv, bias_c0, bias_c1, weight_data_2_clusters)

        """step 1-5 class biases refinement [2021-11-10]
        we can have or not have the repository of defect-labeled X_org
        """
        """1) load repo_c1_data. 
        Note this mechanism should not be taken in the pre-train process. 
        For the test-then-train process, it should not be taken either if CC X_org are imported.
        [2021-11-29] multiple experimental investigations show that such repository mechanism 
        is not helpful with JTI-SDP pf, so I will probably remove this part"""
        use_repo = False  # [record] 2021-11-25, 2022/8/19 to eliminate the related codes
        if use_repo:
            def_data_repo = self.def_data_repo
            nb_def_data = len(np.where(y_obv == 1)[0])
            if nb_def_data < nb_min_c1_refn:
                if def_data_repo.is_repo_empty():  # pre-train, repository is empty
                    CLs_repo_c1 = None
                else:  # repository is not empty
                    pop_X_norm, pop_y_obv, pop_time, nb_pop = def_data_repo.pop_def_data(nb_min_c1_refn - nb_def_data)
                    # derive CLs_pop_X
                    mc_cluster_weights_repo = self.weights_to_points(pop_X_norm)  # step1-3
                    _, CLs_repo_c1 = primary_CLs(pop_y_obv, bias_c0, bias_c1, mc_cluster_weights_repo)  # step1-4

        """2) class bias refinement mechanism"""
        CLs_obv_c1 = CLs_c1[np.where(y_obv == 1)[0]]
        if use_repo and CLs_repo_c1 is not None:  # todo 2021-12-3 to remove
            CLs_obv_c1 = np.concatenate((CLs_obv_c1, CLs_repo_c1))

        """determine the bottom CL of y_obv=1 X_org.
        Goal: the derived CL_c1 of all y_obv=1 equals 1"""
        CLs_c0_refine = np.copy(CLs_c0)
        CLs_c1_refine = np.copy(CLs_c1)
        # when exists defect labelled X_org
        if len(CLs_obv_c1) >= 1:
            # step-1 determine btm_cl_obv_c1
            if len(CLs_obv_c1) > 1:
                # heuristic manner:
                # find the smallest gamma that multiplies std is the bottom CL
                # of all defect X_org acc to the 68-95-99 rule.
                mean_, std_ = np.mean(CLs_obv_c1), np.std(CLs_obv_c1)
                btm_cl_obv_c1 = min_obv_c1 = min(CLs_obv_c1)
                for gamma_ in range(0, 4):
                    if mean_ - std_ * gamma_ <= min_obv_c1:
                        btm_cl_obv_c1 = mean_ - std_ * gamma_
                        continue
                # guarantee btm_cl_obv_c1 to be non-negative
                if btm_cl_obv_c1 < 0:
                    btm_cl_obv_c1 = min_obv_c1
            elif len(CLs_obv_c1) == 1:
                btm_cl_obv_c1 = min_obv_c1 = CLs_obv_c1[0]

            # step-2 find potential defect X_org acc btm_cl_obv_c1
            idx_potential_c1 = np.where(CLs_c1_refine >= btm_cl_obv_c1)[0]
            # re-scale their CLs_c1, see eq.(11) of our ijcnn'22
            # [2022/4/14] Note that we have two variables, namely btm_cl_obv_c1 and  min_obv_c1
            # to perform the re-scale schedule, and it meets that btm_cl_obv_c1 <= min_obv_c1.
            # So not all training X_org with c1 confidence larger than btm_cl_obv_c1 would be
            # rescaled to 1 -- some of them are smaller than 1.
            # So the below codes are not equivalent to
            # >>> CLs_c1_refine[id_potential_c1] = 1
            if len(idx_potential_c1) > 0:
                if min_obv_c1 > 0:
                    CLs_c1_refine[idx_potential_c1] = CLs_c1_refine[idx_potential_c1] / min_obv_c1
                else:
                    CLs_c1_refine[idx_potential_c1] = 1
            # amend those "CLs > 1"
            CLs_c1_refine[np.where(CLs_c1_refine > 1)[0]] = 1
            CLs_c0_refine = 1 - CLs_c1_refine  # sum-1 normalization

        """step 1-6 ultimate CLs of observed yy"""
        option_ = 1
        if option_ == 0:  # original, no CL refinement
            CL_y_obv = [CLs_c0[k] if y_obv[k] == 0 else CLs_c1[k] for k in range(len(y_obv))]
            CL_y_obv = np.asarray(CL_y_obv)
            CL_y_obv[np.where(y_obv == 1)[0]] = 1  # force CLs of defect X_org to the maxima

        elif option_ == 1:  # use this [2021-11-11]
            CL_y_obv = np.asarray(
                [CLs_c0_refine[k] if y_obv[k] == 0 else CLs_c1_refine[k] for k in range(len(y_obv))])

        """[2021-11-28]
        The refinement mechanism is very aggressive towards supporting defect-labeling X_org.
        On the one hand, all actual defect X_org will be identified and being fully used in the model training, 
        and moreover, potentially defect (noisy) X_org can be identified by our core algo.  
        On the other hand, some actual clean X_org would be mis-recognized as "defect", 
        and are filtered out of the training process.
        
        If we do not adopt the refinement mechanism, our core CL algo usually fails recognizing 
        the noisy clean (defect) X_org.
        Considering that usually clean X_org is the majority, it should be OK for us to adopt the refinement mechanism.
        
        In addition, the proposed repo_c1_data should not be taken because it would probably 
        overshoot the defect X_org, which is also demonstrated in our preliminary experimental evaluate.
        
        Therefore, the CL refinement mechanism can only be available in the pre-train process,
        being infeasible in the test-then-train process as only one training X_org being available at each test step.
        Possibly, later when we import CC X_org, the CL refinement mechanism becomes useful 
        for the test-then-train process as there may be more than one training X_org at each test step.
        """
        return CL_y_obv, CLs_c1_refine, use_cluster

    def revise_cluster_info(self, X_norm, y_obv, CLs_est):
        """ Update X_org number of micro-clusters acc. the derived label CLs.
        The reason is that we have already adopted the derived CL info for jit-sdp model learning,
        we should also update related info for the clustering algo accordingly,
        so that label noise can also be excluded from the clustering mechanism to some extent.
        We need a threshold above which we should revise the X_org info of micro-clusters,
        and such threshold is assigned to be the same as in our.partial_fit().

        =======================
        2021-11-16 Liyan created for the CluStream framework
        2021-11-25 copied for DenStream and revised
        """
        self.check_data_dim(X_norm, y_obv)
        CLs_est = np.atleast_1d(CLs_est)

        # update data_stream info for p-clusters (and entire-clusters)
        cluster_labels = self.predict(X_norm)
        for cc in range(len(self.p_micro_clusters)):
            inds = np.where(cluster_labels == cc)[0]
            for _, ind in enumerate(inds):
                if y_obv[ind] == 0 and CLs_est[ind] < 1-self.theta_cl:
                    # p-micro-cluster update
                    self.p_micro_clusters[cc].nb_mc_defect += 1

            assert self.p_micro_clusters[cc].nb_mc_clean >= 0 and \
                   self.p_micro_clusters[cc].nb_mc_defect >= 0

    def plot_cluster(self, X_norm, y, pca, title_info=None, x_lim=None, y_lim=None, plot_clu=True):
        """
        plot in 2D the real jit-sdp X_org and the online clustering process.
        this is to check the core algo.
        para pca: the learnt pca outside the function

        Liyan Song songly@sustech
        Created on 2021-11-24
        """
        # plot all X_org points with the true yy
        X_pca = pca.transform(X_norm)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
        if len(X_pca) <= 20:
            for dd in range(len(X_pca)):
                x_d = X_pca[dd, :]
                plt.text(x_d[0], x_d[1], s=" " + str(dd))
        plt.grid(True)

        if plot_clu:
            # plot p-micro-clusters
            if len(self.p_micro_clusters) > 0:
                theta = np.arange(0, 2 * np.pi, 0.01)
                for ii, p_cluster in enumerate(self.p_micro_clusters):
                    # plot centers
                    center = p_cluster.center()
                    center_pca = pca.transform(center.reshape((1, -1)))[0]
                    plt.scatter(center_pca[0], center_pca[1], marker='x', c='red')
                    # plot mc_radius
                    mc_radius = p_cluster.radius()
                    xc = center_pca[0] + mc_radius * np.cos(theta)
                    yc = center_pca[1] + mc_radius * np.sin(theta)
                    plt.plot(xc, yc, c='red')
                    plt.text(x=center_pca[0], y=center_pca[1], s=" C" + str(ii))
            # plot centers of o-micro-clusters (no radius yet)
            if len(self.o_micro_clusters) > 0:
                for ii, o_cluster in enumerate(self.o_micro_clusters):
                    # plot centers
                    center = o_cluster.center()
                    center_pca = pca.transform(center.reshape((1, -1)))[0]
                    plt.scatter(center_pca[0], center_pca[1], marker='x', c='c')

        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.title(title_info)
        plt.show(block=False)
        # plt.pause(1)
        # plt.close()

    def print_cluster_info(self):
        """ print useful info for DenStream cluster
        2021-11-26 updated
        """
        # p-micro-clusters
        print("\t#p-cluster=%d" % len(self.p_micro_clusters))
        print("\t\t#defect in p-micro-clusters:",
              [p_cluster.nb_mc_defect for p_cluster in self.p_micro_clusters])
        print("\t\t#clean  in p-micro-clusters:",
              [p_cluster.nb_mc_clean for p_cluster in self.p_micro_clusters])

        # o-micro-clusters
        print("\t#o-cluster=%d" % len(self.o_micro_clusters))
        print("\t\t#defect in o-micro-clusters:",
              [o_cluster.nb_mc_defect for o_cluster in self.o_micro_clusters])
        print("\t\t#clean  in o-micro-clusters:",
              [o_cluster.nb_mc_clean for o_cluster in self.o_micro_clusters])

        # radius of p-micro-clusters
        print("\tradius of p-micro-cluster: ",
              [p_cluster.radius() for p_cluster in self.p_micro_clusters])
