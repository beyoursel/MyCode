import numpy as np
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints


class Filter(object):
    def __init__(self, bbox3D, info, ID):
        
        self.initial_pos = bbox3D[:7] # [x, y, z, theta, l, w, h, dx, dy, dz]
        self.time_since_update = 0
        self.id = ID
        self.hits = 1    # number of total hits including the first detection
        self.info = info # other information associated


class KF(Filter):
    def __init__(self, bbox3D, info, ID):
        super().__init__(bbox3D, info, ID)

        self.kf = KalmanFilter(dim_x=10, dim_z=7)

        # state x dimension 10: x, y, z, theta, l, w, h, dx, dy, dz
        # constant velocity model: x' = x + dx, y' = y + dy, z' = z + dz
        # while all others (theta, l, w, h, dx, dy, dz) remain the same

        self.kf.F = np.array([[1,0,0,0,0,0,0,0,0,0], # state transition matrix, dim_x*dim_x状态转移矩阵
                              [0,1,0,0,0,0,0,0,0,0],
                              [0,0,1,0,0,0,0,0,0,1],
		                      [0,0,0,1,0,0,0,0,0,0],  
		                      [0,0,0,0,1,0,0,0,0,0],
		                      [0,0,0,0,0,1,0,0,0,0],
		                      [0,0,0,0,0,0,1,0,0,0],
		                      [0,0,0,0,0,0,0,1,0,0],
		                      [0,0,0,0,0,0,0,0,1,0],
		                      [0,0,0,0,0,0,0,0,0,1]])
        
        # measurement function, dim_z * dim_x, the first 7 dimensions of the measurement correspond to the state
        self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      
		                      [0,1,0,0,0,0,0,0,0,0],
		                      [0,0,1,0,0,0,0,0,0,0],
		                      [0,0,0,1,0,0,0,0,0,0],
		                      [0,0,0,0,1,0,0,0,0,0],
		                      [0,0,0,0,0,1,0,0,0,0],
		                      [0,0,0,0,0,0,1,0,0,0]])
        
        # measurement uncertainty, uncommnent if not super trust the measurement data due to detection noise
        # self.kf.R[0:, 0:] * = 10.

        # initial state uncertainty at time 0
        # Given a single data, the initial velocity if very uncertain, so give a high uncertainty to start
        self.kf.P[7:, 7:] *= 1000. # 初始速度是不确定的，P为协方差矩阵，也叫做预测值和真实值之间的误差协方差
        self.kf.P *= 10.

        # process uncertainty, make the constant velocity part more certain
        self.kf.Q[7:, 7:] *= 0.1 # 噪声变量的协方差矩阵 default=0.01

        # initialize data
        self.kf.x[:7] = self.initial_pos.reshape((7, 1)) # 使用3D目标信息进行初始化

    def compute_innovation_matrix(self):
        """ compute the innovation matrix for association with mahalanobis distance
        """
        return np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T)
    
    def get_velocity(self):
        # return the object velocity in the state
        return self.kf.x[7:]