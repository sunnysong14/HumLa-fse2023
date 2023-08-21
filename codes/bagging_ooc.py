import copy as cp
from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.lazy import KNNADWINClassifier
from skmultiflow.utils.utils import *
from skmultiflow.utils import check_random_state


class OzaBaggingClassifier_OOC(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    # The proposed OOC (ODaSC) in IJCNN2022
    """ OOB bagging ensemble classifier

        Reference:
        .. S. Wang, L. Minku and XX. Yao. "Resampling-based ensemble methods for online class imbalance learning". TKDE, 2015.

        This class is modified by Liyan Song based on Oza's Bagging with small but important changes.
        Specifically, the major update happens in partial_fit().

        NOTE Search by $Liyan$ throughout this script to see the revised places.

        Revised by Liyan Song, songly@sustech.edu.cn
        2021-7-21


        The below is the original documentation.
        ========================================
        Oza Bagging ensemble classifier.

        Parameters
        ----------
        base_estimator: skmultiflow.core.BaseSKMObject or sklearn.BaseEstimator (default=KNNADWINClassifier)
            Each member of the ensemble is an instance of the base estimator.

        n_estimators: int (default=10)
            The size of the ensemble, in other words, how many classifiers to train.

        random_state: int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by `np.random`.

        Raises
        ------
        ValueError: A ValueError is raised if the 'classes' parameter is
        not passed in the first partial_fit call.

        Notes
        -----
        Oza Bagging [1]_ is an ensemble learning method first introduced by Oza and
        Russel's 'Online Bagging and Boosting'. They are an improvement of the
        well known Bagging ensemble method for the batch setting, which in this
        version can effectively handle X_org streams.

        For a traditional Bagging algorithm, adapted for the batch setting, we
        would have M classifiers training on M different datasets, created by
        drawing N XX from the N-sized training set with replacement.

        In the online context, since there is no training dataset, but a stream
        of XX, the drawing of XX with replacement can't be trivially
        executed. The strategy adopted by the Online Bagging algorithm is to
        simulate this task by training each arriving sample K times, which is
        drawn by the binomial distribution. Since we can consider the X_org stream
        to be infinite, and knowing that with infinite XX the binomial
        distribution tends to a Poisson(1) distribution, Oza and Russel found
        that to be a good 'drawing with replacement'.

        References
        ----------
        .. [1] N. C. Oza, “Online Bagging and Boosting,” in 2005 IEEE International Conference on Systems,
           Man and Cybernetics, 2005, vol. 3, no. 3, pp. 2340–2345.

        Examples
        --------
        # >>> # Imports
        # >>> from skmultiflow.meta import OzaBaggingClassifier
        # >>> from skmultiflow.lazy import KNNClassifier
        # >>> from skmultiflow.X_org import SEAGenerator
        # >>> # Setting up the stream
        # >>> stream = SEAGenerator(1, noise_percentage=0.07)
        # >>> # Setting up the OzaBagging classifier to work with KNN as base estimator
        # >>> clf = OzaBaggingClassifier(base_estimator=KNNClassifier(n_neighbors=8, max_window_size=2000, leaf_size=30), n_estimators=2)
        # >>> # Keeping track of sample count and correct prediction count
        # >>> sample_count = 0
        # >>> corrects = 0
        # >>> # Pre training the classifier with 200 XX
        # >>> XX, y = stream.next_sample(200)
        # >>> clf = clf.partial_fit(XX, y, classes=stream.target_values)
        # >>> for i in range(2000):
        # ...     XX, y = stream.next_sample()
        # ...     pred = clf.predict(XX)
        # ...     clf = clf.partial_fit(XX, y)
        # ...     if pred is not None:
        # ...         if y[0] == pred[0]:
        # ...             corrects += 1
        # ...     sample_count += 1
        # >>>
        # >>> # Displaying the evaluate
        # >>> print(str(sample_count) + ' XX analyzed.')
        # 2000 XX analyzed.
        # >>> print('OzaBaggingClassifier performance: ' + str(corrects / sample_count))
        # OzaBagging classifier performance: 0.9095

        """

    def __init__(self, base_estimator=KNNADWINClassifier(), n_estimators=10, random_state=None,
                 theta_imb=0.9, theta_cl=0.8):
        super().__init__()
        # default values
        self.ensemble = None
        self.actual_n_estimators = None
        self.classes = None
        self._random_state = None  # This is the actual random_state object used internally
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.__configure()
        ''' Liyan, 2021-7-22 
        rho0: the size of the clean class (0), corresponding to w- in Table 2 of Shuo paper.
        rho1: the size of the defect-inducing class (+1), corresponding to w+ in Table 2 of Shuo paper.
        theta_imb: w_new = theta_imb * w_old + (1 - theta_imb) * [(x, c)]
        '''
        self.rho0 = 0.5
        self.rho1 = 0.5
        self.theta_imb = theta_imb  # class imbalance update in partial_fit()
        self.theta_cl = theta_cl  # threshold to encode CL with oob in partial_fit()

    def __configure(self):
        if hasattr(self.base_estimator, "reset"):
            self.base_estimator.reset()
        self.actual_n_estimators = self.n_estimators
        self.ensemble = [cp.deepcopy(self.base_estimator) for _ in range(self.actual_n_estimators)]
        self._random_state = check_random_state(self.random_state)

    def reset(self):
        self.__configure()
        return self

    # def partial_fit(self, XX, y, classes=None, sample_weight=None):
    #     """ Liyan:
    #     The only function that had been changed on Oza bagging for OOB
    #
    #     The below is the original documentation.
    #     =========================================
    #
    #     Partially (incrementally) fit the model.
    #
    #     Parameters
    #     ----------
    #     :param XX : numpy.ndarray of shape (n_samples, n_features)
    #         The features to train the model.
    #
    #     :param y: numpy.ndarray of shape (n_samples)
    #         An array-like with the class yy of all XX in XX.
    #
    #     :param classes: numpy.ndarray, optional (default=None)
    #         Array with all possible/known class yy. This is an optional parameter, except
    #         for the first partial_fit call where it is compulsory.
    #
    #     :param sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
    #         Samples weight. If not provided, uniform weights are assumed. Usage varies depending
    #         on the base estimator.
    #
    #     Raises
    #     ------
    #     ValueError
    #         A ValueError is raised if the 'classes' parameter is not passed in the first
    #         partial_fit call, or if they are passed in further calls but differ from
    #         the initial classes list passed.
    #
    #     Returns
    #     -------
    #     OzaBaggingClassifier
    #         self
    #
    #     Notes
    #     -----
    #     Since it's an ensemble learner, if XX and y matrix of more than one
    #     sample are passed, the algorithm will partial fit the model one sample
    #     at a time.
    #
    #     Each sample is trained by each classifier a total of K times, where K
    #     is drawn by a Poisson(1) distribution.
    #
    #     """
    #     if self.classes is None:
    #         if classes is None:
    #             raise ValueError("The first partial_fit call should pass all the classes.")
    #         else:
    #             self.classes = classes
    #
    #     if self.classes is not None and classes is not None:
    #         if set(self.classes) == set(classes):
    #             pass
    #         else:
    #             raise ValueError("The classes passed to the partial_fit function differ from those passed earlier.")
    #
    #     self.__adjust_ensemble_size()
    #     r, _ = get_dimensions(XX)
    #     for j in range(r):
    #         '''Liyan'''
    #         self.rho1 = self.theta_imb * self.rho1 + (1 - self.theta_imb) * (1 if y[j] == 1 else 0)
    #         self.rho0 = self.theta_imb * self.rho0 + (1 - self.theta_imb) * (1 if y[j] == 0 else 0)
    #
    #         for i in range(self.actual_n_estimators):
    #
    #             """Liyan: compute the lambda of Poisson distribution
    #             2021/8/31 found a hug bug in "y[0] == 1 / 0", should be "y[j]"
    #             """
    #             assert 1, "error in 'y[0]', and it should be 'y[j]'. 2021/8/31 found"
    #             if y[0] == 1 and self.rho0 > self.rho1:
    #                 lambda_poisson = self.rho0 / self.rho1
    #             elif y[0] == 0 and self.rho0 < self.rho1:
    #                 lambda_poisson = self.rho1 / self.rho0
    #             else:
    #                 lambda_poisson = 1
    #
    #             """Liyan: get K from Poisson(lambda_poisson) distribution"""
    #             k = self._random_state.poisson(lambda_poisson)  # core revision from Oza_bagging to OOB
    #             # k = self._random_state.poisson()  # original oza code
    #
    #             if k > 0:
    #                 for b in range(k):
    #                     self.ensemble[i].partial_fit([XX[j]], [y[j]], classes, sample_weight)
    #     return self

    def partial_fit(self, X, y, CLs_np, classes=None, sample_weight=None):
        """
        Liyan Song: songly@sustech.edu.cn
        2021-8-31   my main revision on oob
        2021-12-14  Shuxian finds a bug making nb_tree useless as all trees employ exactly the same training X_org
        """
        if self.classes is None:
            if classes is None:
                raise ValueError("The first partial_fit call should pass all the classes.")
            else:
                self.classes = classes

        if self.classes is not None and classes is not None:
            if set(self.classes) == set(classes):
                pass
            else:
                raise ValueError("The classes passed to the partial_fit function differ from those passed earlier.")

        self.__adjust_ensemble_size()

        """Liyan revision is mainly in the following part [2021/8/30]"""
        r, _ = get_dimensions(X)  # r: X_org size
        y = y.reshape(r, )
        CLs_np = CLs_np.reshape(r, )

        for j in range(r):  # for each X_org point
            """1) track class imbalance in the online manner"""
            self.rho1 = self.theta_imb * self.rho1 + (1 - self.theta_imb) * (1 if y[j] == 1 else 0)
            self.rho0 = self.theta_imb * self.rho0 + (1 - self.theta_imb) * (1 if y[j] == 0 else 0)

            """2) compute improved OOB's lambda, see [Shuo TKDE2015]"""
            if y[j] == 1 and self.rho0 > self.rho1:
                lambda_poisson = self.rho0 / self.rho1
            elif y[j] == 0 and self.rho0 < self.rho1:
                lambda_poisson = self.rho1 / self.rho0
            else:
                lambda_poisson = 1

            """3) incorporate CL's alpha with lambda of OOB
            # 2021-11-30 tend to choose-3 
            """
            opt_cl = 3
            lambda_alpha = lambda_poisson
            if opt_cl == 0:  # hard threshold [2021-8-31]
                if CLs_np[j] < self.theta_cl:
                    lambda_alpha = lambda_poisson * CLs_np[j]

            elif opt_cl == 1:  # ReLu threshold, 2021-11-30, debug and get this
                if CLs_np[j] < self.theta_cl:
                    lambda_alpha = lambda_poisson * CLs_np[j] / self.theta_cl

            elif opt_cl == 2:  # rescale (buggy implement of Rely), 2021/11/30 found
                lambda_alpha = lambda_poisson * CLs_np[j] / self.theta_cl

            elif opt_cl == 3:  # non-linear sigmoid-like rescale, 2021/11/30
                """this option is designed with the below rules:
                when cl == threshold:   cl_rescale = 1
                when cl < threshold:    cl_rescale < 1
                when cl > threshold:    cl_rescale > 1"""
                lambda_alpha = lambda_poisson * math.exp(CLs_np[j]-self.theta_cl)
                """E.g. 
                threshold=0.8, cl_rescale in [0.449, 1.22] 
                    - min(cl_rescale) = exp(1-0.8) = 1.22, max(cl_rescale) = exp(0-0.8) = 0.449
                threshold=0.9, cl_rescale in [0.4065, 1.1051]
                """

            """4) upgrade to oob from oza"""
            for i in range(self.actual_n_estimators):  # all base learners
                k = self._random_state.poisson(lambda_alpha)
                # k = self._random_state.poisson()  # save of oza
                if k > 0:
                    for b in range(k):
                        self.ensemble[i].partial_fit([X[j]], [y[j]], classes, sample_weight)
        return self

    def __adjust_ensemble_size(self):
        if len(self.classes) != len(self.ensemble):
            if len(self.classes) > len(self.ensemble):
                for i in range(len(self.ensemble), len(self.classes)):
                    self.ensemble.append(cp.deepcopy(self.base_estimator))
                    self.actual_n_estimators += 1

    def predict(self, X):
        """ Predict classes for the passed X_org.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of X_org XX to predict the class yy for.

        Returns
        -------
        A numpy.ndarray with all the predictions for the XX in XX.

        Notes
        -----
        The predict function will average the predictions from all its learners
        to find the most likely prediction for the sample matrix XX.

        """
        # Debug Liyan on 2021-12-7
        # without this, an error arises without any warning
        if np.ndim(X) == 1:
            X = X.reshape((1, -1))

        r, c = get_dimensions(X)
        proba = self.predict_proba(X)
        predictions = []
        if proba is None:
            return None
        for i in range(r):
            predictions.append(np.argmax(proba[i]))
        return np.asarray(predictions)

    def predict_proba(self, X):
        """ Estimates the probability of each sample in XX belonging to each of the class-yy.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The matrix of XX one wants to predict the class probabilities for.

        Returns
        -------
        A numpy.ndarray of shape (n_samples, n_labels), in which each outer entry is associated with the XX entry of the
        same index. And where the list in index [i] contains len(self.target_values) elements, each of which represents
        the probability that the i-th sample of XX belongs to a certain class-label.

        Raises
        ------
        ValueError: A ValueError is raised if the number of classes in the base_estimator
        learner differs from that of the ensemble learner.

        """
        proba = []
        r, c = get_dimensions(X)
        try:
            for i in range(self.actual_n_estimators):
                partial_proba = self.ensemble[i].predict_proba(X)
                if len(partial_proba[0]) > max(self.classes) + 1:
                    raise ValueError("The number of classes in the base learner is larger than in the ensemble.")

                if len(proba) < 1:
                    for n in range(r):
                        proba.append([0.0 for _ in partial_proba[n]])

                for n in range(r):
                    for l in range(len(partial_proba[n])):
                        try:
                            proba[n][l] += partial_proba[n][l]
                        except IndexError:
                            proba[n].append(partial_proba[n][l])
        except ValueError:
            return np.zeros((r, 1))
        except TypeError:
            return np.zeros((r, 1))

        # normalizing probabilities
        sum_proba = []
        for l in range(r):
            sum_proba.append(np.sum(proba[l]))
        aux = []
        for i in range(len(proba)):
            if sum_proba[i] > 0.:
                aux.append([x / sum_proba[i] for x in proba[i]])
            else:
                aux.append(proba[i])
        return np.asarray(aux)
