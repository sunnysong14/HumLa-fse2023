import numpy as np
import math
# from scipy.spatial import distance


class MicroCluster:
    def __init__(self, lambd, creation_time, nb_points=0):
        self.lambd = lambd
        self.decay_factor = 2 ** (-lambd)
        self.mean = 0
        self.variance = 0
        self.sum_of_weights = 0
        self.creation_time = creation_time

        """additional attributes of micro-clusters"""
        self.nb_mc_clean = 0
        self.nb_mc_defect = 0
        self.nb_mc_unlabelled = 0

    def insert_sample(self, one_sample, new_y, weight):
        # update data_stream number
        nb_new_clean = np.count_nonzero(new_y == 0)
        nb_new_defect = np.count_nonzero(new_y == 1)
        nb_new_unlabelled = 1 - nb_new_defect - nb_new_clean
        self.nb_mc_clean += nb_new_clean
        self.nb_mc_defect += nb_new_defect
        self.nb_mc_unlabelled += nb_new_unlabelled

        # insert this sample
        if self.sum_of_weights != 0:
            # Update sum of weights
            old_sum_of_weights = self.sum_of_weights
            new_sum_of_weights = old_sum_of_weights * self.decay_factor + weight

            # Update mean
            old_mean = self.mean
            new_mean = old_mean + \
                (weight / new_sum_of_weights) * (one_sample - old_mean)

            # Update variance
            old_variance = self.variance
            new_variance = old_variance * ((new_sum_of_weights - weight)
                                           / old_sum_of_weights) \
                + weight * (one_sample - new_mean) * (one_sample - old_mean)
            self.mean = new_mean
            self.variance = new_variance
            self.sum_of_weights = new_sum_of_weights
        else:  # the 1st one_sample of this micro-cluster
            self.mean = one_sample
            self.sum_of_weights = weight

    def radius(self):
        if self.sum_of_weights > 0:
            return np.linalg.norm(np.sqrt(self.variance / self.sum_of_weights))
        else:
            return float('nan')

    def center(self):
        return self.mean

    def weight(self):
        """for an existing p-micro-cluster Cj, if no new point is merged into it,
        the weight of Cj will decay gradually. If the weight is below beta*mu,
        it means that Cj becomes an outlier, and should be deleted from the p-cluster list"""
        return self.sum_of_weights

    def __copy__(self):
        new_micro_cluster = MicroCluster(self.lambd, self.creation_time)
        new_micro_cluster.sum_of_weights = self.sum_of_weights
        new_micro_cluster.variance = self.variance
        new_micro_cluster.mean = self.mean
        return new_micro_cluster

    def class_bias(self, prob_mc, prob_data):
        """ Compute micro-cluster's confidence level (CL).
        They are the bias of a micro-cluster towards class label 0 or label 1.
        NOTE It is possible that "cl(C0) + cl(C1) != 0".

        :param prob_mc: probability of a micro-cluster of being label 0 or 1.
        :param prob_data: probability of the entire X_org of being label 0 or 1.
        Note that prob_mc and prob_data should denote the same class label simultaneously.

        :return cl_mc: CL of the bias toward class label 0 or 1 for the micro-cluster
        """
        assert 0 <= prob_mc <= 1 and 0 <= prob_data <= 1
        direct_distance = self.directed_distance_probs(prob_mc, prob_data)
        cl_mc = prob_mc * (1 + direct_distance)  # range ~ P(C)*[1/2, 3/2)
        return cl_mc

    def directed_distance_probs(self, prob_mc, prob_data):
        """ Liyan Song
        Directed distance between prob_mc (P_C0) and prob_data (P_D0).

        :param prob_mc: probability in the micro-cluster for a certain label 0/1
        :param prob_data: probability in the while X_org for a certain label 0/1
        :return direct_distance: directed distance

        Liyan Song: songly@sustech.edu.cn
        2021-8-23
        2021-11-24 copied from CluStream to DenStream
        """

        # direct_distance should be in (-0.5, 0.5]
        if prob_mc == prob_data:  # incl. a special case that "prob_mc == prob_data == 0"
            direct_distance = 0
        else:
            direct_distance = 0.5 * (prob_mc - prob_data) / (prob_mc + prob_data)
        assert -1 / 2 <= direct_distance < 1 / 2
        return direct_distance

    def weight_to_1point(self, data_1point, temperature=10):
        """ Compute the weight of micro-cluster Cj to determine CL of X_org Xi with two factors:
        1) similarity between Xi and Cj,
        2) concept drift metric

        2021-11-10  latest update in CluStream
        2021-11-24  copied for DenStream from CluStream and revised
        """

        """1) distance betw X_org Xi and micro-cluster Cj
        """
        eu_distance = self.distance_to_point(data_1point)
        option_ = 1  # we choose this
        if option_ == 0:
            dis_X_C = eu_distance
        elif option_ == 1:  # use-this [2021-11-10]
            dis_X_C = max((0, eu_distance - self.radius()))
        elif option_ == 2:
            dis_X_C = eu_distance / self.radius()

        """convert to similarity [2021-11-11]
        hyper-para temperature T: larger T can enlarge the differentiation among data_stream"""
        T = temperature  # 10 by default, temperature
        similarity = math.exp(-dis_X_C*T)
        assert 0 <= similarity <= 1, "similarity="+str(similarity)

        """2) obsolescence of micro-cluster Cj
        The computation here would be very different from that for CluStream
        since DenStream does not have method such as get_relevancestamp(),
        and it uses dif mechanism tracing time decay by micro_cluster.weight().
        We should incorporate our core CL algo with DenStream's mechanism"""
        T2 = temperature
        up_to_date = np.tanh(self.weight()/T2)  # range in (0,1)
        # print("up_to_date", up_to_date)  # tmp

        return similarity, up_to_date

    def distance_to_point(self, sample):
        # Liyan created on 2021-11-26, DenStream._get_nearest_micro_cluster()
        return np.linalg.norm(self.center() - sample)
