import math
import scipy
import numpy as np

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

def build_gap(x, lamda=1.0):
    # return (1 + math.exp(-1)) / (1 + math.exp(-x))
    return (1 + lamda * math.exp(-1)) / (1 + lamda * math.exp(-x))

class ExtendKalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, w, h, vx, vy, vw, vh, ax, aw

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self, srate, stdp, stdv, stda, adjusted_gate):
        ndim, edim, dt = 4, 2, build_gap(srate)
        self.edim = 2

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim + edim, 2 * ndim + edim)
        
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # * Extended Kalman Filter
        self._motion_mat[ndim, -2] = dt
        self._motion_mat[ndim + 2, -1] = dt
        self._motion_mat[0, -2] = 0.5 * dt ** 2
        self._motion_mat[2, -1] = 0.5 * dt ** 2
            
        self._update_mat = np.eye(ndim, 2 * ndim + edim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.

        self.kalman_gain = adjusted_gate

        self._std_weight_position = stdp
        self._std_weight_velocity = stdv
        self._std_weight_acceleration = stda

        self._std_weight_position_q = stdp * self.kalman_gain
        self._std_weight_velocity_q = stdv * self.kalman_gain
        self._std_weight_acceleration_q = stda * self.kalman_gain

        self._std_weight_position_r = stdp * (2.0 - self.kalman_gain)
        self._std_weight_velocity_r = stdv * (2.0 - self.kalman_gain)
        self._std_weight_acceleration_r = stda * (2.0 - self.kalman_gain)

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean_acc = np.zeros(self.edim)
        mean = np.r_[mean_pos, mean_vel, mean_acc]

        std = [2 * self._std_weight_position * measurement[3],
               2 * self._std_weight_position * measurement[3],
               2 * self._std_weight_position * measurement[3],
               2 * self._std_weight_position * measurement[3],

               10 * self._std_weight_velocity * measurement[3],
               10 * self._std_weight_velocity * measurement[3],
               10 * self._std_weight_velocity * measurement[3],
               10 * self._std_weight_velocity * measurement[3],
               
               2 * self._std_weight_acceleration * measurement[3],
               2 * self._std_weight_acceleration * measurement[3]]

        covariance = np.diag(np.square(std))
        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        # TODO: 观测 噪声矩阵R
        std = [self._std_weight_position_r * mean[3],
               self._std_weight_position_r * mean[3],
               self._std_weight_position_r * mean[3],
               self._std_weight_position_r * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        # TODO: 观测 噪声矩阵R
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        
        return mean, covariance + innovation_cov

    def update_alpha(self, mean, covariance, measurement, alpha):
        std_pos = [
            self._std_weight_position_q * mean[3],
            self._std_weight_position_q * mean[3],
            self._std_weight_position_q * mean[3],
            self._std_weight_position_q * mean[3]]
        std_vel = [
            self._std_weight_velocity_q * mean[3],
            self._std_weight_velocity_q * mean[3],
            self._std_weight_velocity_q * mean[3],
            self._std_weight_velocity_q * mean[3]]
        std_acc = [self._std_weight_acceleration_q * mean[3],
                   self._std_weight_acceleration_q * mean[3]]

        std = [self._std_weight_position_r * mean[3],
               self._std_weight_position_r * mean[3],
               self._std_weight_position_r * mean[3],
               self._std_weight_position_r * mean[3]]
        innovation_cov = np.diag(np.square(std))

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel, std_acc]))

        innovation = np.dot(self._update_mat, mean) - measurement
        sigma_V = (alpha / (1 + alpha)) * np.linalg.multi_dot((innovation, innovation.T))

        M = np.linalg.multi_dot((self._update_mat, self._motion_mat, covariance, self._motion_mat.T, self._update_mat.T))
        N = sigma_V - np.linalg.multi_dot((self._update_mat, motion_cov, self._update_mat.T)) - innovation_cov

        Nt, Mt = np.trace(N), np.trace(M)

        if Nt > Mt:
            alpha = min(Nt / Mt, 1.01)
        return alpha

    def predict(self, mean, covariance, alpha):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        # TODO: 系统 噪声矩阵Q
        std_pos = [
            self._std_weight_position_q * mean[3],
            self._std_weight_position_q * mean[3],
            self._std_weight_position_q * mean[3],
            self._std_weight_position_q * mean[3]]
        std_vel = [
            self._std_weight_velocity_q * mean[3],
            self._std_weight_velocity_q * mean[3],
            self._std_weight_velocity_q * mean[3],
            self._std_weight_velocity_q * mean[3]]
        std_acc = [self._std_weight_acceleration_q * mean[3],
                   self._std_weight_acceleration_q * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel, std_acc]))
        # TODO: 系统 噪声矩阵Q

        #mean = np.dot(self._motion_mat, mean)
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) * alpha + motion_cov

        return mean, covariance

    def multi_predict(self, mean, covariance, alpha):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        last_covariance = covariance.copy()

        # TODO: 系统 噪声矩阵Q
        std_pos = [
            self._std_weight_position_q * mean[:, 3],
            self._std_weight_position_q * mean[:, 3],
            self._std_weight_position_q * mean[:, 3],
            self._std_weight_position_q * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity_q * mean[:, 3],
            self._std_weight_velocity_q * mean[:, 3],
            self._std_weight_velocity_q * mean[:, 3],
            self._std_weight_velocity_q * mean[:, 3]]
        std_acc = [self._std_weight_acceleration_q * mean[:, 3],
                   self._std_weight_acceleration_q * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel, std_acc]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)
        # TODO: 系统 噪声矩阵Q

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        alpha = alpha[:, np.newaxis, np.newaxis].repeat(10, axis=1).repeat(10, axis=2)

        covariance = np.dot(left, self._motion_mat.T) * alpha + motion_cov

        return mean, covariance, last_covariance

    def update(self, mean, covariance, measurement, alpha):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        # 通过估计值和观测值估计最新结果
        alpha = self.update_alpha(mean, covariance, measurement, alpha)
        covariance = self.transform_covariance(mean, covariance, alpha)

        # 将均值和协方差映射到检测空间，得到 Hx' 和 S
        projected_mean, projected_cov = self.project(mean, covariance)

        # 矩阵分解
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)

        # 计算卡尔曼增益K
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T

        # z - Hx'
        innovation = measurement - projected_mean

        # x = x' + Ky
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        
        # P = (I - KH)P'
        new_covariance = (covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T)))

        return new_mean, new_covariance, alpha
    
    def transform_covariance(self, mean, covariance, alpha):
        # TODO: 系统 噪声矩阵Q
        std_pos = [
            self._std_weight_position_q * mean[3],
            self._std_weight_position_q * mean[3],
            self._std_weight_position_q * mean[3],
            self._std_weight_position_q * mean[3]]
        std_vel = [
            self._std_weight_velocity_q * mean[3],
            self._std_weight_velocity_q * mean[3],
            self._std_weight_velocity_q * mean[3],
            self._std_weight_velocity_q * mean[3]]
        std_acc = [self._std_weight_acceleration_q * mean[3],
                   self._std_weight_acceleration_q * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel, std_acc]))
        # TODO: 系统 噪声矩阵Q
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) * alpha + motion_cov

        return covariance

    def gating_distance(self, mean, covariance, measurements,
                    only_position=False):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha

def xywh2xywh(mean):
    x_c, y, w, h = mean
    return np.array([x_c - w / 2, y - h, w, h])