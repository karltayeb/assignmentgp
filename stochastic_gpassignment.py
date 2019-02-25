import numpy as np
import tensorflow as tf

from gpflow.models import GPModel
from gpflow import likelihoods
from gpflow import settings
from gpflow.decors import autoflow
from gpflow.decors import params_as_tensors
from gpflow.mean_functions import Zero
from gpflow.params import DataHolder, Parameter, Minibatch
from gpflow import transforms, kullback_leiblers, features
from gpflow.conditionals import conditional


class GaussianSAGP(GPModel):
    """
    Sparse Variational GP regression. The key reference is
    ::
      @inproceedings{titsias2009variational,
        title={Variational learning of inducing variables in
               sparse Gaussian processes},
        author={Titsias, Michalis K},
        booktitle={International Conference onum_i
                   Artificial Intelligence and Statistics},
        pages={567--574},
        year={2009}
      }
    """

    def __init__(self, X, Y, W, kern, idx=None, feat=None, Z=None,
                 mean_function=None, q_diag=False, whiten=False,
                 q_mu=None, q_sqrt=None,
                 minibatch_size=None, num_latent=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        kern, mean_function are appropriate GPflow objects
        This method only works with a Gaussian likelihood.
        """
        num_data = X.shape[0]

        if minibatch_size is None:
            X = DataHolder(X, fix_shape=True)
            Y = DataHolder(Y, fix_shape=True)

        else:
            X = Minibatch(X, batch_size=minibatch_size, seed=0)
            Y = Minibatch(Y, batch_size=minibatch_size, seed=0)

        # init the super class
        likelihood = likelihoods.Gaussian()
        num_latent = W.shape[1]
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function,
                         num_latent=num_latent, **kwargs)

        if minibatch_size is not None:
            idx = Minibatch(np.arange(num_data), batch_size=minibatch_size, seed=0, dtype=np.int32)

        self.idx = idx
        self.W = Parameter(W, trainable=False)
        self.K = self.W.shape[1]
        self.W_prior = Parameter(np.ones(self.K) / self.K, trainable=False)
        self.num_data = num_data
        self.feature = features.inducingpoint_wrapper(feat, Z)

        self.minibatch_size = minibatch_size
        self.q_diag, self.whiten = q_diag, whiten

        # init variational parameters
        num_inducing = len(self.feature)
        self._init_variational_parameters(
            num_inducing, q_mu, q_sqrt, q_diag)

        #self.L = tf.cholesky(
        #    features.Kuu(self.feature, self.kern, jitter=settings.jitter))

    def _init_variational_parameters(self, num_inducing, q_mu, q_sqrt, q_diag):
        """
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the routine
        initializes them, their shape depends on `num_inducing` and `q_diag`.
        Note: most often the comments refer to the number of observations (=output dimensions) with P,
        number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multioutput kernels are used, this can change.
        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically refered to as M.
        :param q_mu: np.array or None
            Mean of the variational Gaussian posterior. If None the function will initialise
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt: np.array or None
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function will initialise `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag: bool
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """
        q_mu = np.zeros((num_inducing, self.num_latent)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=settings.float_type)  # M x P

        if q_sqrt is None:
            if self.q_diag:
                self.q_sqrt = Parameter(np.ones((num_inducing, self.num_latent), dtype=settings.float_type),
                                        transform=transforms.positive)  # M x P
            else:
                q_sqrt = np.array([np.eye(num_inducing, dtype=settings.float_type) for _ in range(self.num_latent)])
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.LowerTriangular(num_inducing, self.num_latent))  # P x M x M
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                self.num_latent = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.positive)  # M x L/P
            else:
                assert q_sqrt.ndim == 3
                self.num_latent = q_sqrt.shape[0]
                num_inducing = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.LowerTriangular(num_inducing, self.num_latent))  # L/P x M x M

    @params_as_tensors
    def build_prior_KL(self):
        if self.whiten:
            K = None
        else:
            K = features.Kuu(
                self.feature, self.kern,
                jitter=settings.numerics.jitter_level)  # (P x) x M x M

        return kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt, K)

    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see the associated
        SGPR notebook.
        """
        Y = self.Y
        X = self.X
        idx = self.idx

        if self.minibatch_size is not None:
            W = tf.gather(self.W, idx)
            W = tf.reshape(W, [-1, self.K])
            W = normalize(W)
        else:
            W = normalize(self.W)  # N x K

        ND = tf.cast(tf.size(Y), settings.float_type)
        D = tf.cast(tf.shape(Y)[1], settings.float_type)
        sigma2 = self.likelihood.variance

        # Get kernel terms
        # Expand if necessary?
        Kdiag = self.kern.Kdiag(X, full_output_cov=False)
        Kux = features.Kuf(self.feature, self.kern, X)
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)

        # compute statistics (potentially on minibatch)
        psi0 = self._psi0(Kdiag)
        psi1 = self._psi1(Kux, W)  # K x M x N
        psi2 = self._psi2(Kux, W)  # K x M x M

        L = tf.cholesky(Kuu)
        A = tf.matrix_triangular_solve(L, psi1)  # K x M x N
        a = tf.matrix_triangular_solve(
            L, tf.transpose(self.q_mu)[:, :, None])  # K x M x 1
        mean = tf.matmul(A, a, transpose_a=True)

        tmp1 = tf.matrix_triangular_solve(L, psi2)
        B = tf.matrix_triangular_solve(L, tf.transpose(tmp1, perm=[0, 2, 1]))

        tmp2 = tf.matrix_triangular_solve(L, self.q_sqrt)
        C = tf.matmul(tmp2, tmp2, transpose_b=True)

        # compute KL
        KL1 = self.build_prior_KL()
        KL2 = tf.reduce_sum(W * (
            tf.log(W) - tf.log(self.W_prior)[None]))

        # compute log marginal bound
        bound = -0.5 * ND * tf.log(2 * np.pi * sigma2)
        bound += -0.5 * tf.reduce_sum(tf.square(Y)) / sigma2
        bound += tf.reduce_sum(Y * mean) / sigma2
        bound += -0.5 * tf.reduce_sum(
            tf.matmul(a, tf.matmul(B, a), transpose_a=True)) / sigma2
        bound += -0.5 * D * (psi0 - tf.reduce_sum(tf.matrix_diag_part(B)))
        bound += -0.5 * tf.reduce_sum(tf.einsum('kmp,kpm->km', B, C))

        bound -= KL2

        if self.minibatch_size is not None:
            scale = self.num_data / self.minibatch_size
            bound *= scale

        bound -= KL1
        return bound

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, full_output_cov=False):
        mu, var = conditional(
            Xnew, self.feature, self.kern,
            self.q_mu, q_sqrt=self.q_sqrt, full_cov=full_cov,
            white=self.whiten, full_output_cov=full_output_cov)
        return mu + self.mean_function(Xnew), var

    @autoflow()
    @params_as_tensors
    def compute_qu(self):
        """
        Computes the mean and variance of q(u), the variational distribution on
        inducing outputs. SVGP with this q(u) should predict identically to
        SGPR.
        :return: mu, A
        """
        Y = self.Y
        X = self.X
        idx = self.idx

        Kux = features.Kuf(self.feature, self.kern, X)
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)
        W = normalize(self.W)  # N x K

        psi1 = self._psi1(Kux, W)  # K x M x N
        psi2 = self._psi2(Kux, W)  # K x M x M

        Sig = Kuu + (self.likelihood.variance ** -1) * psi2
        Sig_sqrt = tf.cholesky(Sig)

        Sig_sqrt_Kuu = tf.matrix_triangular_solve(Sig_sqrt, Kuu)

        A = tf.matmul(Sig_sqrt_Kuu, Sig_sqrt_Kuu, transpose_a=True)

        tmp = tf.matrix_triangular_solve(Sig_sqrt, psi1)
        P = tf.einsum('kmn,nd->kmd', tmp, Y)
        mu = tf.matmul(Sig_sqrt_Kuu, P, transpose_a=True) * self.likelihood.variance ** -1.0

        return mu[:, :, 0], tf.cholesky(A)

    @autoflow()
    @params_as_tensors
    def update_W(self):
        X = self.X
        Y = self.Y
        idx = self.idx

        Kdiag = self.kern.Kdiag(X, full_output_cov=False)
        Kux = features.Kuf(self.feature, self.kern, X)
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)
        L = tf.cholesky(Kuu)

        sigma2 = self.likelihood.variance

        A = tf.cholesky_solve(L, Kux)  # K x M x N
        mean = tf.matmul(A, tf.transpose(
            self.q_mu)[:, :, None], transpose_a=True)
        err = (Y - mean)

        reg1 = tf.reduce_sum(
            tf.pow(tf.matmul(A, self.q_sqrt, transpose_a=True), 2), 2)

        reg2 = tf.transpose(Kdiag) - tf.einsum('kmn,kmn->kn', A, Kux)
        logW = -0.5 * tf.log(2 * np.pi * sigma2) \
            - 0.5 * tf.reduce_sum(tf.pow(err, 2), 2) / sigma2 \
            - 0.5 * reg1 / sigma2 - 0.5 * reg2 / sigma2 + tf.log(self.W_prior)[:, None]

        logW = logW - tf.reduce_logsumexp(logW, axis=0, keepdims=True)
        return logW, idx


    @autoflow((settings.float_type, [None, None]), (settings.float_type, [None, None]))
    @params_as_tensors
    def update_W_external(self, X, Y):
        Kdiag = self.kern.Kdiag(X, full_output_cov=False)
        Kux = features.Kuf(self.feature, self.kern, X)
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)

        # Copy this into blocks for each dimension
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)
        L = tf.cholesky(Kuu)

        sigma2 = self.likelihood.variance

        A = tf.cholesky_solve(L, Kux)  # K x M x N
        mean = tf.matmul(A, tf.transpose(self.q_mu)[:, :, None], transpose_a=True)
        err = (Y - mean)

        reg1 = tf.reduce_sum(
            tf.pow(tf.matmul(A, self.q_sqrt, transpose_a=True), 2), 2)
        reg2 = tf.transpose(Kdiag) - tf.einsum('kmn,kmn->kn', A, Kux)

        logW = -0.5 * tf.log(2 * np.pi * sigma2) \
            - 0.5 * tf.reduce_sum(tf.pow(err, 2), 2) / sigma2 \
            - 0.5 * reg1 / sigma2 - 0.5 * reg2 / sigma2 + tf.log(self.W_prior)[:, None]

        logW = logW - tf.reduce_logsumexp(logW, axis=0, keepdims=True)
        return tf.transpose(logW)

    @params_as_tensors
    def _psi0(self, Kdiag):
        return tf.reduce_sum(Kdiag)

    @params_as_tensors
    def _psi1(self, Kux, W):
        psi1 = Kux * tf.transpose(W)[:, None]
        return psi1  # K x M x N

    @params_as_tensors
    def _psi2(self, Kux, W):
        psi2 = tf.einsum('kmn,kpn->kmp', Kux, Kux * tf.transpose(W)[:, None])
        return psi2  # K x M x M


class GaussianSMAGP(GPModel):
    """
    Sparse Variational GP regression. The key reference is
    ::
      @inproceedings{titsias2009variational,
        title={Variational learning of inducing variables in
               sparse Gaussian processes},
        author={Titsias, Michalis K},
        booktitle={International Conference onum_i
                   Artificial Intelligence and Statistics},
        pages={567--574},
        year={2009}
      }
    """

    def __init__(self, X, Y, W1, W2, kern, idx=None, W1_idx=None, W2_idx=None, feat=None, Z=None,
                 mean_function=None, q_diag=False, whiten=False,
                 q_mu=None, q_sqrt=None,
                 minibatch_size=None, num_latent=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        kern, mean_function are appropriate GPflow objects
        This method only works with a Gaussian likelihood.
        """
        num_data = X.shape[0]

        if minibatch_size is None:
            X = DataHolder(X, fix_shape=True)
            Y = DataHolder(Y, fix_shape=True)

            if W1_idx is not None:
                W1_idx = DataHolder(W1_idx, fix_shape=True)
            if W2_idx is not None:
                W2_idx = DataHolder(W2_idx, fix_shape=True)

        else:
            X = Minibatch(X, batch_size=minibatch_size, seed=0)
            Y = Minibatch(Y, batch_size=minibatch_size, seed=0)
            idx = Minibatch(np.arange(num_data), batch_size=minibatch_size, seed=0, dtype=np.int32)

            if W1_idx is not None:
                W1_idx = Minibatch(
                    W1_idx, batch_size=minibatch_size, seed=0, dtype=np.int32)
            if W2_idx is not None:
                W2_idx = Minibatch(
                    W2_idx, batch_size=minibatch_size, seed=0, dtype=np.int32)

        # init the super class
        likelihood = likelihoods.Gaussian()
        num_latent = W1.shape[1] * W2.shape[1]
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function,
                         num_latent=num_latent, **kwargs)

        self.idx = idx
        self.W1_idx = W1_idx
        self.W2_idx = W2_idx

        self.K1 = W1.shape[1]
        self.W1 = Parameter(W1, trainable=False, dtype=settings.float_type)
        self.W1_prior = Parameter(np.ones(self.K1) / self.K1, trainable=False)

        self.K2 = W2.shape[1]
        self.W2 = Parameter(W2, trainable=False, dtype=settings.float_type)
        self.W2_prior = Parameter(np.ones(self.K2) / self.K2, trainable=False)

        self.num_data = num_data
        self.feature = features.inducingpoint_wrapper(feat, Z)

        self.minibatch_size = minibatch_size
        self.q_diag, self.whiten = q_diag, whiten

        # init variational parameters
        num_inducing = len(self.feature)
        self._init_variational_parameters(
            num_inducing, q_mu, q_sqrt, q_diag)

    def _init_variational_parameters(self, num_inducing, q_mu, q_sqrt, q_diag):
        """
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the routine
        initializes them, their shape depends on `num_inducing` and `q_diag`.
        Note: most often the comments refer to the number of observations (=output dimensions) with P,
        number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multioutput kernels are used, this can change.
        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically refered to as M.
        :param q_mu: np.array or None
            Mean of the variational Gaussian posterior. If None the function will initialise
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt: np.array or None
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function will initialise `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag: bool
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """
        q_mu = np.zeros((num_inducing, self.num_latent)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=settings.float_type)  # M x P

        if q_sqrt is None:
            if self.q_diag:
                self.q_sqrt = Parameter(np.ones((num_inducing, self.num_latent), dtype=settings.float_type),
                                        transform=transforms.positive)  # M x P
            else:
                q_sqrt = np.array([np.eye(num_inducing, dtype=settings.float_type) for _ in range(self.num_latent)])
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.LowerTriangular(num_inducing, self.num_latent))  # P x M x M
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                self.num_latent = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.positive)  # M x L/P
            else:
                assert q_sqrt.ndim == 3
                self.num_latent = q_sqrt.shape[0]
                num_inducing = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.LowerTriangular(num_inducing, self.num_latent))  # L/P x M x M

    @params_as_tensors
    def build_prior_KL(self):
        if self.whiten:
            K = None
        else:
            K = features.Kuu(
                self.feature, self.kern,
                jitter=settings.numerics.jitter_level)  # (P x) x M x M

        return kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt, K)

    @params_as_tensors
    def build_prior_assignment_KL(self, W1_idx, W2_idx):
        if W1_idx is None:
            W1 = self.W1
        else:
            W1 = tf.gather(self.W1, tf.unique(W1_idx)[0])
        
        if W2_idx is None:
            W2 = self.W2
        else:
            W2 = tf.gather(self.W2, tf.unique(W2_idx)[0])

        KL = 0

        if True:
            KL += tf.reduce_sum(normalize(W1) * (
                tf.log(normalize(W1)) - tf.log(self.W1_prior)[None])) \

        if True:
            KL += tf.reduce_sum(normalize(W2) * (
                tf.log(normalize(W2)) - tf.log(self.W2_prior)[None]))

        return KL

    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see the associated
        SGPR notebook.
        """
        Y = self.Y
        X = self.X
        idx = self.idx

        W1_idx = self.W1_idx
        W2_idx = self.W2_idx

        if W1_idx is None:
            W1_idx = idx
        if W2_idx is None:
            W2_idx = idx

        if self.minibatch_size is not None:
            W1 = tf.gather(self.W1, W1_idx)
            W1 = tf.reshape(W1, [-1, self.K1])
            W1 = normalize(W1)

            W2 = tf.gather(self.W2, W2_idx)
            W2 = tf.reshape(W2, [-1, self.K2])
            W2 = normalize(W2)

        else:
            W1 = normalize(self.W1)  # N x K1
            if W1_idx is not None:
                W1 = tf.gather(W1, W1_idx)
            W2 = normalize(self.W2)  # N x K2
            if W2_idx is not None:
                W2 = tf.gather(W2, W2_idx)

        ND = tf.cast(tf.size(Y), settings.float_type)
        D = tf.cast(tf.shape(Y)[1], settings.float_type)
        sigma2 = self.likelihood.variance

        # Get kernel terms
        # Expand if necessary?
        Kdiag = self.kern.Kdiag(X, full_output_cov=False)
        Kux = features.Kuf(self.feature, self.kern, X)
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)

        W = _expand_W(W1, W2)
        # compute statistics (potentially on minibatch)
        psi0 = self._psi0(Kdiag)
        psi1 = self._psi1(Kux, W)  # K x M x N
        psi2 = self._psi2(Kux, W)  # K x M x M

        L = tf.cholesky(Kuu)

        A = tf.matrix_triangular_solve(L, psi1)  # K x M x N
        a = tf.matrix_triangular_solve(
            L, tf.transpose(self.q_mu)[:, :, None])  # K x M x 1
        mean = tf.matmul(A, a, transpose_a=True)

        tmp1 = tf.matrix_triangular_solve(L, psi2)
        B = tf.matrix_triangular_solve(L, tf.transpose(tmp1, perm=[0, 2, 1]))

        tmp2 = tf.matrix_triangular_solve(L, self.q_sqrt)
        C = tf.matmul(tmp2, tmp2, transpose_b=True)

        # compute KL
        KL1 = self.build_prior_KL()
        KL2 = self.build_prior_assignment_KL(W1_idx, W2_idx)

        # compute log marginal bound
        bound = -0.5 * ND * tf.log(2 * np.pi * sigma2)
        bound += -0.5 * tf.reduce_sum(tf.square(Y)) / sigma2
        bound += tf.reduce_sum(Y * mean) / sigma2
        bound += -0.5 * tf.reduce_sum(
            tf.matmul(a, tf.matmul(B, a), transpose_a=True)) / sigma2
        bound += -0.5 * D * (psi0 - tf.reduce_sum(tf.matrix_diag_part(B)))
        bound += -0.5 * tf.reduce_sum(tf.einsum('kmp,kpm->km', B, C))

        if self.minibatch_size is not None:
            scale = self.num_data / self.minibatch_size
            bound *= scale

        bound -= KL2
        bound -= KL1

        return bound

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, full_output_cov=False):
        mu, var = conditional(
            Xnew, self.feature, self.kern,
            self.q_mu, q_sqrt=self.q_sqrt, full_cov=full_cov,
            white=self.whiten, full_output_cov=full_output_cov)
        return mu + self.mean_function(Xnew), var

    @autoflow()
    @params_as_tensors
    def compute_qu(self):
        """
        Computes the mean and variance of q(u), the variational distribution on
        inducing outputs. SVGP with this q(u) should predict identically to
        SGPR.
        :return: mu, A
        """
        Y = self.Y
        X = self.X
        idx = self.idx

        W1_idx = self.W1_idx
        W2_idx = self.W2_idx

        if W1_idx is None:
            W1_idx = idx
        if W2_idx is None:
            W2_idx = idx

        if self.minibatch_size is not None:
            W1 = tf.gather(self.W1, W1_idx)
            W1 = tf.reshape(W1, [-1, self.K1])
            W1 = normalize(W1)

            W2 = tf.gather(self.W2, W2_idx)
            W2 = tf.reshape(W2, [-1, self.K2])
            W2 = normalize(W2)

        else:
            W1 = normalize(self.W1)  # N x K1
            if W1_idx is not None:
                W1 = tf.gather(W1, W1_idx)
            W2 = normalize(self.W2)  # N x K2
            if W2_idx is not None:
                W2 = tf.gather(W2, W2_idx)

        Kux = features.Kuf(self.feature, self.kern, X)
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)

        W = _expand_W(W1, W2)
        psi1 = self._psi1(Kux, W)  # K x M x N
        psi2 = self._psi2(Kux, W)  # K x M x M

        Sig = Kuu + (self.likelihood.variance ** -1) * psi2
        Sig_sqrt = tf.cholesky(Sig)

        Sig_sqrt_Kuu = tf.matrix_triangular_solve(Sig_sqrt, Kuu)

        A = tf.matmul(Sig_sqrt_Kuu, Sig_sqrt_Kuu, transpose_a=True)

        tmp = tf.matrix_triangular_solve(Sig_sqrt, psi1)
        P = tf.einsum('kmn,nd->kmd', tmp, Y)
        mu = tf.matmul(Sig_sqrt_Kuu, P, transpose_a=True) * self.likelihood.variance ** -1.0

        return mu[:, :, 0], tf.cholesky(A)

    @autoflow((settings.float_type, [None, None]),
        (settings.float_type, [None, None]),
        (tf.int32, [None]),
        (tf.int32, [None]))
    @params_as_tensors
    def compute_qu_external(self, X, Y, W1_idx, W2_idx):
        """
        Computes the mean and variance of q(u), the variational distribution on
        inducing outputs. SVGP with this q(u) should predict identically to
        SGPR.
        :return: mu, A
        """
        Kux = features.Kuf(self.feature, self.kern, X)
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)

        W1 = tf.gather(normalize(self.W1), W1_idx)  # N x K
        W2 = tf.gather(normalize(self.W2), W2_idx)

        W = _expand_W(W1, W2)
        psi1 = self._psi1(Kux, W)  # K x M x N
        psi2 = self._psi2(Kux, W)  # K x M x M

        Sig = Kuu + (self.likelihood.variance ** -1) * psi2
        Sig_sqrt = tf.cholesky(Sig)

        Sig_sqrt_Kuu = tf.matrix_triangular_solve(Sig_sqrt, Kuu)

        A = tf.matmul(Sig_sqrt_Kuu, Sig_sqrt_Kuu, transpose_a=True)

        tmp = tf.matrix_triangular_solve(Sig_sqrt, psi1)
        P = tf.einsum('kmn,nd->kmd', tmp, Y)
        mu = tf.matmul(Sig_sqrt_Kuu, P, transpose_a=True) * self.likelihood.variance ** -1.0

        return mu[:, :, 0], tf.cholesky(A)

    @autoflow()
    @params_as_tensors
    def update_W(self):
        Y = self.Y
        X = self.X
        idx = self.idx

        W1_idx = self.W1_idx
        W2_idx = self.W2_idx

        if W1_idx is None:
            W1_idx = idx
        if W2_idx is None:
            W2_idx = idx

        Kdiag = self.kern.Kdiag(X, full_output_cov=False)
        Kux = features.Kuf(self.feature, self.kern, X)
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)

        # Copy this into blocks for each dimension
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)
        L = tf.cholesky(Kuu)

        sigma2 = self.likelihood.variance

        A = tf.cholesky_solve(L, Kux)  # K x M x N
        mean = tf.matmul(A, tf.transpose(self.q_mu)[:, :, None], transpose_a=True)
        err = (Y - mean)

        reg1 = tf.reduce_sum(
            tf.pow(tf.matmul(A, self.q_sqrt, transpose_a=True), 2), 2)
        reg2 = tf.transpose(Kdiag) - \
            tf.einsum('kmn,kmn->kn', A, Kux)

        logW = -0.5 * tf.log(2 * np.pi * sigma2) \
            - 0.5 * tf.reduce_sum(tf.pow(err, 2), 2) / sigma2 \
            - 0.5 * reg1 / sigma2 - 0.5 * reg2 / sigma2

        logW = tf.reshape(logW, [self.K1, self.K2, -1]) \
            + tf.transpose(tf.log(self.W1_prior))[:, None, None] \
            + tf.transpose(tf.log(self.W2_prior))[None :, None]

        # compute new W1
        W2 = normalize(self.W2)
        if W2_idx is not None:
            W2 = tf.gather(W2, W2_idx)

        logW1 = tf.reduce_sum(
            logW * tf.transpose(W2)[None, :, :],
            axis=1)
        logW1 = logW1 - tf.reduce_logsumexp(logW1, axis=0, keepdims=True)

        # group by index
        logW1_parts = tf.dynamic_partition(
            tf.transpose(logW1), W1_idx, num_partitions=self.W1.shape[0])
        logW1 = tf.stack([
            tf.reduce_sum(part, axis=0) for part in logW1_parts])
        logW1 = logW1 + self.W1_prior[None]
        logW1 = logW1 - tf.reduce_logsumexp(logW1, axis=1, keepdims=True)

        # compute new W2
        W1 = normalize(logW1)
        if W1_idx is not None:
            W1 = tf.gather(W1, W1_idx)

        logW2 = tf.reduce_sum(
            logW * tf.transpose(W1)[:, None, :],
            axis=0)
        logW2 = logW2 - tf.reduce_logsumexp(logW2, axis=0, keepdims=True)

        # group by index
        logW2_parts = tf.dynamic_partition(
            tf.transpose(logW2), W2_idx, num_partitions=self.W2.shape[0])
        logW2 = tf.stack([
            tf.reduce_sum(part, axis=0) for part in logW2_parts])
        logW2 = logW2 + self.W2_prior[None]
        logW2 = logW2 - tf.reduce_logsumexp(logW2, axis=1, keepdims=True)
        return logW1, logW2

    @autoflow((settings.float_type, [None, None]), (settings.float_type, [None, None]), (tf.int32, [None]), (tf.int32, [None]))
    @params_as_tensors
    def update_W_external(self, X, Y, W1_idx, W2_idx):
        Kdiag = self.kern.Kdiag(X, full_output_cov=False)
        Kux = features.Kuf(self.feature, self.kern, X)
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)

        # Copy this into blocks for each dimension
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)
        L = tf.cholesky(Kuu)

        sigma2 = self.likelihood.variance

        A = tf.cholesky_solve(L, Kux)  # K x M x N
        mean = tf.matmul(A, tf.transpose(self.q_mu)[:, :, None], transpose_a=True)
        err = (Y - mean)

        reg1 = tf.reduce_sum(
            tf.pow(tf.matmul(A, self.q_sqrt, transpose_a=True), 2), 2)
        reg2 = tf.transpose(Kdiag) - \
            tf.einsum('kmn,kmn->kn', A, Kux)

        logW = -0.5 * tf.log(2 * np.pi * sigma2) \
            - 0.5 * tf.reduce_sum(tf.pow(err, 2), 2) / sigma2 \
            - 0.5 * reg1 / sigma2 - 0.5 * reg2 / sigma2
        logW = tf.reshape(logW, [self.K1, self.K2, -1])

        # compute new W1
        W2 = tf.gather(normalize(self.W2), W2_idx)
        logW1 = tf.reduce_sum(
            logW * tf.transpose(W2)[None, :, :],
            axis=1)
        logW1 = logW1 - tf.reduce_logsumexp(logW1, axis=0, keepdims=True)

        # group by index
        logW1_parts = tf.dynamic_partition(
            tf.transpose(logW1), W1_idx, num_partitions=self.W1.shape[0])
        logW1 = tf.stack([
            tf.reduce_sum(part, axis=0) for part in logW1_parts])
        logW1 = logW1 + self.W1_prior[None]
        logW1 = logW1 - tf.reduce_logsumexp(logW1, axis=1, keepdims=True)

        # compute new W2
        W1 = tf.gather(normalize(logW1), W1_idx)
        logW2 = tf.reduce_sum(
            logW * tf.transpose(W1)[:, None, :],
            axis=0)
        logW2 = logW2 - tf.reduce_logsumexp(logW2, axis=0, keepdims=True)

        # group by index
        logW2_parts = tf.dynamic_partition(
            tf.transpose(logW2), W2_idx, num_partitions=self.W2.shape[0])
        logW2 = tf.stack([
            tf.reduce_sum(part, axis=0) for part in logW2_parts])
        logW2 = logW2 + self.W2_prior[None]
        logW2 = logW2 - tf.reduce_logsumexp(logW2, axis=1, keepdims=True)
        return logW1, logW2

    @params_as_tensors
    def _psi0(self, Kdiag):
        return tf.reduce_sum(Kdiag)

    @params_as_tensors
    def _psi1(self, Kux, W):
        psi1 = Kux * tf.transpose(W)[:, None]
        return psi1  # K x M x N

    @params_as_tensors
    def _psi2(self, Kux, W):
        psi2 = tf.einsum('kmn,kpn->kmp', Kux, Kux * tf.transpose(W)[:, None])
        return psi2  # K x M x M


class SMAGP(GPModel):
    """
    Sparse Variational GP regression. The key reference is
    ::
      @inproceedings{titsias2009variational,
        title={Variational learning of inducing variables in
               sparse Gaussian processes},
        author={Titsias, Michalis K},
        booktitle={International Conference onum_i
                   Artificial Intelligence and Statistics},
        pages={567--574},
        year={2009}
      }
    """

    def __init__(self, X, Y, W1, W2, kern, likelihood,
                 idx=None, W1_idx=None, W2_idx=None, feat=None, Z=None,
                 mean_function=None, q_diag=False, whiten=False,
                 q_mu=None, q_sqrt=None,
                 minibatch_size=None, num_latent=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        kern, mean_function are appropriate GPflow objects
        This method only works with a Gaussian likelihood.
        """
        num_data = X.shape[0]

        if minibatch_size is None:
            X = DataHolder(X, fix_shape=True)
            Y = DataHolder(Y, fix_shape=True)

            if W1_idx is not None:
                W1_idx = DataHolder(W1_idx, fix_shape=True)
            if W2_idx is not None:
                W2_idx = DataHolder(W2_idx, fix_shape=True)

        else:
            X = Minibatch(X, batch_size=minibatch_size, seed=0)
            Y = Minibatch(Y, batch_size=minibatch_size, seed=0)
            idx = Minibatch(np.arange(num_data), batch_size=minibatch_size, seed=0, dtype=np.int32)

            if W1_idx is not None:
                W1_idx = Minibatch(
                    W1_idx, batch_size=minibatch_size, seed=0, dtype=np.int32)
            if W2_idx is not None:
                W2_idx = Minibatch(
                    W2_idx, batch_size=minibatch_size, seed=0, dtype=np.int32)

        # init the super class
        num_latent = W1.shape[1] * W2.shape[1]
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function,
                         num_latent=num_latent, **kwargs)

        self.idx = idx
        self.W1_idx = W1_idx
        self.W2_idx = W2_idx

        self.K1 = W1.shape[1]
        self.W1 = Parameter(W1, trainable=False, dtype=settings.float_type)
        self.W1_prior = Parameter(np.ones(self.K1) / self.K1, trainable=False)

        self.K2 = W2.shape[1]
        self.W2 = Parameter(W2, trainable=False, dtype=settings.float_type)
        self.W2_prior = Parameter(np.ones(self.K2) / self.K2, trainable=False)

        self.num_data = num_data
        self.feature = features.inducingpoint_wrapper(feat, Z)

        self.minibatch_size = minibatch_size
        self.q_diag, self.whiten = q_diag, whiten

        # init variational parameters
        num_inducing = len(self.feature)
        self._init_variational_parameters(
            num_inducing, q_mu, q_sqrt, q_diag)

    def _init_variational_parameters(self, num_inducing, q_mu, q_sqrt, q_diag):
        """
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the routine
        initializes them, their shape depends on `num_inducing` and `q_diag`.
        Note: most often the comments refer to the number of observations (=output dimensions) with P,
        number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multioutput kernels are used, this can change.
        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically refered to as M.
        :param q_mu: np.array or None
            Mean of the variational Gaussian posterior. If None the function will initialise
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt: np.array or None
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function will initialise `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag: bool
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """
        q_mu = np.zeros((num_inducing, self.num_latent)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=settings.float_type)  # M x P

        if q_sqrt is None:
            if self.q_diag:
                self.q_sqrt = Parameter(np.ones((num_inducing, self.num_latent), dtype=settings.float_type),
                                        transform=transforms.positive)  # M x P
            else:
                q_sqrt = np.array([np.eye(num_inducing, dtype=settings.float_type) for _ in range(self.num_latent)])
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.LowerTriangular(num_inducing, self.num_latent))  # P x M x M
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                self.num_latent = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.positive)  # M x L/P
            else:
                assert q_sqrt.ndim == 3
                self.num_latent = q_sqrt.shape[0]
                num_inducing = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.LowerTriangular(num_inducing, self.num_latent))  # L/P x M x M

    @params_as_tensors
    def build_prior_KL(self):
        if self.whiten:
            K = None
        else:
            K = features.Kuu(
                self.feature, self.kern,
                jitter=settings.numerics.jitter_level)  # (P x) x M x M

        return kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt, K)

    @params_as_tensors
    def build_prior_assignment_KL(self, W1_active, W2_active):
        W1 = tf.gather(self.W1, W1_active)
        W2 = tf.gather(self.W2, W2_active)

        KL = 0

        if True:
            KL += tf.reduce_sum(normalize(W1) * (
                tf.log(normalize(W1)) - tf.log(self.W1_prior)[None])) \

        if True:
            KL += tf.reduce_sum(normalize(W2) * (
                tf.log(normalize(W2)) - tf.log(self.W2_prior)[None]))

        return KL

    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see the associated
        SGPR notebook.
        """
        Y = self.Y
        X = self.X
        idx = self.idx

        W1_idx = self.W1_idx
        W2_idx = self.W2_idx

        if W1_idx is None:
            W1_idx = idx
        if W2_idx is None:
            W2_idx = idx

        if self.minibatch_size is not None:
            W1 = tf.gather(self.W1, W1_idx)
            W1 = tf.reshape(W1, [-1, self.K1])
            W1 = normalize(W1)

            W2 = tf.gather(self.W2, W2_idx)
            W2 = tf.reshape(W2, [-1, self.K2])
            W2 = normalize(W2)

        else:
            W1 = normalize(self.W1)  # N x K1
            if W1_idx is not None:
                W1 = tf.gather(W1, W1_idx)
            W2 = normalize(self.W2)  # N x K2
            if W2_idx is not None:
                W2 = tf.gather(W2, W2_idx)

        ND = tf.cast(tf.size(Y), settings.float_type)
        D = tf.cast(tf.shape(Y)[1], settings.float_type)
        sigma2 = self.likelihood.variance

        # Get kernel terms
        # Expand if necessary?
        Kdiag = self.kern.Kdiag(X, full_output_cov=False)
        Kux = features.Kuf(self.feature, self.kern, X)
        Kuu = features.Kuu(self.feature, self.kern, jitter=settings.jitter)

        W = _expand_W(W1, W2)

       # compute KL
        KL1 = self.build_prior_KL()
        KL2 = self.build_prior_assignment_KL(tf.unique(W1_idx)[0], tf.unique(W2_idx)[0])

        fmean, fvar = self._build_predict(X, full_cov=False, full_output_cov=False)
        var_exp = self.likelihood.variational_expectations(fmean, fvar, Y)
        scale = self.num_data / self.minibatch_size

        bound = tf.reduce_sum(W * var_exp)
        bound *= scale
        bound -= KL1 + KL2
        return bound

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, full_output_cov=False):
        mu, var = conditional(
            Xnew, self.feature, self.kern,
            self.q_mu, q_sqrt=self.q_sqrt, full_cov=full_cov,
            white=self.whiten, full_output_cov=full_output_cov)
        return mu + self.mean_function(Xnew), var


def normalize(W, epsilon=1e-3):
    expW = tf.exp(W) + epsilon
    return expW / tf.reduce_sum(expW, axis=1)[:, None]


def _expand_W(W1, W2):
    K1 = W1.shape[1]
    K2 = W2.shape[1]

    if not isinstance(K1, int):
        K1 = K1.value
        K2 = K2.value

    W = []
    for i in range(K1):
        for j in range(K2):
            W.append(W1[:, int(i)] * W2[:, int(j)])

    return tf.transpose(tf.stack(W))


