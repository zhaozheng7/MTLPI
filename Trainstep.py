import tensorflow as tf
from scipy.optimize import minimize_scalar, minimize
from tensorflow import keras
import numpy as np
from keras import backend as K
from cagrad import CAGrad
from PCGrad import PCGrad


class PI_Controler(keras.Model):
    def init_arguments(self, method, coverage_rate, loss_name):
        self.method = method
        self.coverage_rate = coverage_rate
        self.lamda = 5
        self.loss_name = loss_name
        """print 当前参数"""
        print("method:", self.method)
        print("coverage_rate:", self.coverage_rate)
        print("loss_name:", self.loss_name)
        # Define selective loss function

    def selective_up(self, y_true, y_pred):
        """
        Equation 11b, 分为两部分，1/c*求和（（上界-真实值）**2）*ki-------------------1/c*求和（（真实值-下界）**2）*ki
        用于计算选择性宽度上采样损失
        首先，根据输入的y_true和y_pred，计算出两个指示符ub_ind和lb_ind。
        ub_ind表示预测值在真实值之上（即sigmoid(y_pred[:,1] - y_true[:,0]) > 0.5）的指示符。
        lb_ind表示真实值在预测值之上（即sigmoid(y_true[:,0] - y_pred[:,0]) > 0.5）的指示符。 这两个指示符将用于确定哪些样本应该被用于计算损失。
        然后，通过将ub_ind和lb_ind相乘，得到一个指示符indicator，该指示符同时满足ub_ind和lb_ind条件的样本。
        接下来，计算宽度上采样损失width_up_loss，即预测值的第二列与真实值的第一列之间的差的平方。
        仅计算indicator为真的样本的width_up_loss，通过将indicator与width_up_loss相乘实现。
        最后，将计算出的width_up_loss进行平均，得到最终的损失值。通过添加K.epsilon()防止除以零
        :param y_true: 真实值
        :param y_pred: 预测值，第一列为预测下界，第二列为预测上界
        :return:上宽度损失
        """
        ub_ind = K.cast(K.greater(K.sigmoid(y_pred[:, 1] - y_true[:, 0]), 0.5), K.floatx())
        lb_ind = K.cast(K.greater(K.sigmoid(y_true[:, 0] - y_pred[:, 0]), 0.5), K.floatx())
        indicator = tf.multiply(ub_ind, lb_ind)
        ###
        width_up_loss = K.square(y_pred[:, 1] - y_true[:, 0])
        # Only compute selected ones
        width_up_loss = tf.math.multiply(indicator, width_up_loss)
        width_up_loss = (K.sum(width_up_loss) + K.epsilon()) / (K.sum(indicator) + K.epsilon())
        return width_up_loss

    def selective_low(self, y_true, y_pred):
        """
        Equation 11b 1/c*求和（（真实值-下界）**2）*ki
        用于计算选择性宽度下采样损失
        最终的
        :param y_true:
        :param y_pred: 预测值，第一列为预测下界，第二列为预测上界
        :return: 下宽度损失
        """
        ub_ind = K.cast(K.greater(K.sigmoid(y_pred[:, 1] - y_true[:, 0]), 0.5), K.floatx())
        lb_ind = K.cast(K.greater(K.sigmoid(y_true[:, 0] - y_pred[:, 0]), 0.5), K.floatx())
        indicator = tf.multiply(ub_ind, lb_ind)
        width_low_loss = K.square(y_true[:, 0] - y_pred[:, 0])
        # Only compute selected ones
        width_low_loss = tf.math.multiply(indicator, width_low_loss)
        width_low_loss = (K.sum(width_low_loss) + K.epsilon()) / (K.sum(indicator) + K.epsilon())
        return width_low_loss

    # Upper penalty function (upper_bound >= y_true)
    # 定义上界惩罚函数（上界>=真实值）equation 11d miss gamma
    def up_penalty(self, y_true, y_pred):
        """
        Equation 11d miss gamma
        该函数用于计算预测值与真实值之间的惩罚损失。其中，y_true为真实标签的tensor，y_pred为预测标签的tensor。
        函数首先计算y_true的第二列与y_pred的第二列之间的差值，并对差值进行取最大值操作，然后对所有元素求和得到惩罚损失值。最后返回该损失值。
        :param y_true: 真实值
        :param y_pred: 预测值，第一列为预测下界，第二列为预测上界
        :return:覆盖率
        """
        up_penalty_loss = K.sum(K.maximum([0.0], (y_true[:, 1] - y_pred[:, 1])))
        return up_penalty_loss

    # Lower penalty function (lower_bound =< y_true)
    def low_penalty(self, y_true, y_pred):
        """
        Equation 11e miss gamma
        :param y_true:
        :param y_pred:
        :return:
        """
        low_penalty_loss = K.sum(K.maximum([0.0], y_pred[:, 0] - y_true[:, 1]))
        return low_penalty_loss

    # Coverage penalty function
    def coverage_penalty(self, y_true, y_pred):
        """
        Equation 11f , Calculate the gamma use in up_penalty and low_penalty
        :param y_true:
        :param y_pred:
        :return:gamma use in up_penalty and low_penalty
        """
        ub_ind = K.cast(K.greater(K.sigmoid(y_pred[:, 1] - y_true[:, 0]), 0.5), K.floatx())
        lb_ind = K.cast(K.greater(K.sigmoid(y_true[:, 0] - y_pred[:, 0]), 0.5), K.floatx())
        coverage_value = K.mean(tf.multiply(ub_ind, lb_ind))
        # return self.lamda * (K.exp(K.maximum(0.0, self.coverage_rate - coverage))-1)
        return self.lamda * (K.maximum(0.0, self.coverage_rate - coverage_value))

    # Calculate the coverage
    def coverage(self, y_true, y_pred):
        """
        Equation 11g
        :param y_true:
        :param y_pred:
        :return:
        """
        coverage_value = K.cast(K.mean((y_pred[:, 1] >= y_true[:, 0]) & (y_true[:, 0] >= y_pred[:, 0])), K.floatx())
        return coverage_value

    # Calculate Mean predicition interval width(mpiw)
    def mpiw(self, y_true, y_pred):
        res = K.cast(K.mean(K.square(y_pred[:, 0] - y_pred[:, 1])), K.floatx())
        return res

    """
    这里使用新的损失函数来自2024年的IEEE论文
    """

    def abs_selective_up(self, y_true, y_pred):
        """
        分步计算
        :param y_true: 真实值
        :param y_pred: 预测值，0下界，1上界
        :return: 返回上宽度指标损失
        """
        width_up_loss = K.abs(y_pred[:, 1] - y_true[:, 0])
        # width_up_loss = K.abs(y_pred[:, 1] - y_pred[:, 2])
        width_up_loss = K.mean(width_up_loss)
        return width_up_loss

    def abs_selective_low(self, y_true, y_pred):
        """
        分步计算
        :param y_true: 真实值
        :param y_pred: 预测值，0下界，1上界
        :return: 返回下宽度指标损失
        """
        width_low_loss = K.abs(y_true[:, 0] - y_pred[:, 0])
        # width_low_loss = K.abs(y_pred[:, 2] - y_pred[:, 0])
        width_low_loss = K.mean(width_low_loss)
        return width_low_loss

    def abs_mpiw(self, y_true, y_pred):
        """
        分步计算
        :param y_true: 真实值
        :param y_pred: 预测值，0下界，1上界
        :return: 返回MPIW指标损失
        """
        width_up_loss = K.abs(y_pred[:, 1] - y_true[:, 0])
        width_up_loss = K.mean(width_up_loss)
        width_low_loss = K.abs(y_true[:, 0] - y_pred[:, 0])
        width_low_loss = K.mean(width_low_loss)
        return width_up_loss + width_low_loss

    def abs_ki_selective_up(self, y_true, y_pred):
        ub_ind = K.cast(K.greater(K.sigmoid(y_pred[:, 1] - y_true[:, 0]), 0.5), K.floatx())
        lb_ind = K.cast(K.greater(K.sigmoid(y_true[:, 0] - y_pred[:, 0]), 0.5), K.floatx())
        indicator = tf.multiply(ub_ind, lb_ind)
        ###
        width_up_loss = K.abs(y_pred[:, 1] - y_true[:, 0])
        # Only compute selected ones
        width_up_loss = tf.math.multiply(indicator, width_up_loss)
        width_up_loss = (K.sum(width_up_loss) + K.epsilon()) / (K.sum(indicator) + K.epsilon())
        return width_up_loss

    def abs_ki_selective_low(self, y_true, y_pred):
        ub_ind = K.cast(K.greater(K.sigmoid(y_pred[:, 1] - y_true[:, 0]), 0.5), K.floatx())
        lb_ind = K.cast(K.greater(K.sigmoid(y_true[:, 0] - y_pred[:, 0]), 0.5), K.floatx())
        indicator = tf.multiply(ub_ind, lb_ind)
        width_low_loss = K.abs(y_true[:, 0] - y_pred[:, 0])
        # Only compute selected ones
        width_low_loss = tf.math.multiply(indicator, width_low_loss)
        width_low_loss = (K.sum(width_low_loss) + K.epsilon()) / (K.sum(indicator) + K.epsilon())
        return width_low_loss

    def abs_rmpiw_loss(self, y_true, y_pred):
        """
        使用rmpiw进行宽度损失函数构造
        :param y_true:
        :param y_pred:
        :return:
        """
        width_up_loss = K.abs(y_pred[:, 1] - y_true[:, 0])
        width_up_loss = K.mean(width_up_loss)
        width_low_loss = K.abs(y_true[:, 0] - y_pred[:, 0])
        width_low_loss = K.mean(width_low_loss)
        r = K.max(y_true[:, 0]) - K.min(y_true[:, 0])
        return (width_up_loss + width_low_loss) / r

    def dual_picp(self, y_true, y_pred):
        """
        使用dual论文中的picp计算方法
        :param y_true:
        :param y_pred:
        :return:
        """
        y_l = y_pred[:, 0]
        y_u = y_pred[:, 1]
        y_o = y_pred[:, 2]
        picp_now = K.cast(K.mean((y_pred[:, 1] >= y_true[:, 0]) & (y_true[:, 0] >= y_pred[:, 0])), K.floatx())
        c = self.coverage_rate - picp_now
        eta_ = c * 0.001
        cs = tf.math.reduce_max(tf.abs(y_o - y_true[:, 2]))  # cs 为点估计和真实值差距最大的地方
        Constraints = K.exp(K.abs(K.mean(-y_u + y_true[:, 2]) + cs)) + K.exp(K.abs(K.mean(-y_true[:, 2] + y_l) + cs))
        return eta_ * Constraints

    def picp_term(self, y_true, y_pred):
        """
        计算PICP的正则项
        :param y_true: 真实值
        :param y_pred: 预测值，0下界，1上界
        :return: PICP的term
        """
        picp_now = tf.reduce_mean(tf.cast(
            tf.logical_and(tf.less_equal(y_pred[:, 0], y_pred[:, 2]), tf.less_equal(y_pred[:, 2], y_pred[:, 1])),
            tf.float32))
        constraint_loss = 8888 * tf.maximum(0.0, self.coverage_rate - picp_now)
        loss_picp = K.sum(K.maximum([0.0], (y_true[:, 1] - y_pred[:, 1])))
        return constraint_loss

    def mse_penalty(self, y_true, y_pred):
        """
        点估计
        :param y_true: 真实值
        :param y_pred: 预测值，0下界，1上界, 2预测
        :return: 惩罚损失
        """
        # mse_loss = K.square(y_true[:,2] - y_pred[:,2])
        # mse_loss = K.mean(mse_loss)
        return tf.losses.mean_squared_error(y_true[:, 0], y_pred[:, 2])

    def cov_loss(self, y_true, y_pred):
        """
        用于覆盖率满足预设值的损失函数，源自于2024的IEEE dual论文
        :param y_true:真实值
        :param y_pred:预测值，0下界，1上界，2预测值
        :return:覆盖率损失
        """
        y_l = y_pred[:, 0]
        y_u = y_pred[:, 1]
        y_o = y_pred[:, 2]
        cs = K.max((K.abs(y_true[:, 0] - y_o)).detach())
        du = K.mean(y_u - y_true[:, 0])
        dl = K.mean(-y_l + y_true[:, 0])
        penalty = K.exp(cs - du) + K.exp(cs - dl)
        return penalty

    def adaptive_hyper(self, y_true, y_pred):
        """
        自适应lambda值
        :param y_true:0下界，1上界，2预测值
        :param y_pred:0下界，1上界，2预测值
        :return:
        """
        # calculate the coverage
        yita = 0.001

        picp_now = tf.logical_and(tf.less_equal(y_pred[:, 0], y_pred[:, 2]), tf.less_equal(y_pred[:, 2], y_pred[:, 1]))
        picp_now=tf.reduce_mean(tf.cast(picp_now, tf.float32))
        c = self.coverage_rate - picp_now

        return yita * c

    def calculate_piad(self, y_true, y_pred):
        """
        暂时用不到
        计算偏差信息
        :param y_true:真实值
        :param y_pred:预测值
        :return: 偏差信息损失函数
        """
        y_l = y_pred[:, 0]
        y_u = y_pred[:, 1]
        y_o = y_pred[:, 2]
        piad = K.zeros_like(y_l)
        piad = tf.where(y_true[:, 0] <= y_l, y_l - y_true[:, 0], piad)
        piad = tf.where(y_true[:, 0] >= y_u, y_true[:, 0] - y_u, piad)
        l_piad = K.sum(piad)
        return l_piad

    def dual_loss(self, y_true, y_pred, C):
        """
        loss from 2024 dual paper
        :param y_true:0下界，1上界，2预测值
        :param y_pred:0下界，1上界，2预测值
        :return:
        """
        y_l = y_pred[:, 0]  # 下界
        y_u = y_pred[:, 1]  # 上界
        y_o = y_pred[:, 2]  # 预测值
        y_true = y_true[:, 0]  # 真实值()
        beta_ = [1+C, 0.99]
        lambda1, lambda2 = beta_
        MSE = K.mean((y_o - y_true) ** 2)  # Calculate MSE
        # cs = tf.math.reduce_max(tf.abs(y_o - y_true))  # cs 为点估计和真实值差距最大的地方
        # DualAQD reported in the paper // torch.clamp的作用是若第一个参数小于第二个参数，则返回第二个参数，否则返回第一个参数
        MPIW_p = K.mean(K.abs(y_u - y_o) + K.abs(y_o - y_l))  # Calculate MPIW_penalty
        Constraints = (
                    K.exp(K.mean(-y_u + y_o) + K.max(K.abs(y_o - y_true))) +
                    K.exp(K.mean(-y_o + y_l) + K.max(K.abs(y_o - y_true))) +
                    K.exp(K.mean(-y_u + y_l))
        )
        return MPIW_p + lambda1 * Constraints + lambda2 * MSE

    def qd_loss(self, y_true, y_pred):
        """
        use QD loss by Tea
        :param y_true:
        :param y_pred:
        :return:
        """
        # hyperparameters
        # set to 4.0 for naval, 40.0 for protein, 30.0 for wine, and 6.0 for yacht
        lambda_ = 40.  # lambda in loss fn
        alpha_ = 1. - self.coverage_rate  # capturing (1-alpha)% of samples
        soften_ = 160.
        n_ = 100  # batch size
        y_true = y_true[:, 2]
        y_l = y_pred[:, 0]
        y_u = y_pred[:, 1]
        K_HU = tf.maximum(0., tf.sign(y_u - y_true))
        K_HL = tf.maximum(0., tf.sign(y_true - y_l))
        K_H = tf.multiply(K_HU, K_HL)
        K_SU = tf.sigmoid(soften_ * (y_u - y_true))
        K_SL = tf.sigmoid(soften_ * (y_true - y_l))
        K_S = tf.multiply(K_SU, K_SL)
        MPIW_c = tf.reduce_sum(tf.multiply((y_u - y_l), K_H)) / tf.reduce_sum(K_H)
        PICP_H = tf.reduce_mean(K_H)
        PICP_S = tf.reduce_mean(K_S)
        Loss_S = MPIW_c + lambda_ * 10 / (alpha_ * (1 - alpha_)) * (tf.maximum(0., (1 - alpha_) - PICP_S))
        return Loss_S

    def qd_plus_loss(self, y_true, y_pred):
        """
        use qd plus by snm paper
        :param y_true:
        :param y_pred:
        :return:
        """
        y_true = y_true[:, 0]
        y_l = y_pred[:, 0]
        y_u = y_pred[:, 1]
        y_o = y_pred[:, 2]
        # Separate hyperparameters
        beta_ = [0.9974, 0.51]
        lambda_1, lambda_2 = beta_
        ksi = 10  # According to the QD+ paper
        soften_ = 160.
        MSE = K.mean((y_o - y_true) ** 2)  # Calculate MSE
        alpha_ = 1 - self.coverage_rate
        # Calculate soft captured vector, MPIW, and PICP
        K_SU = K.sigmoid(soften_ * (y_u - y_true))
        K_SL = K.sigmoid(soften_ * (y_true - y_l))
        K_S = tf.multiply(K_SU, K_SL)
        MPIW_c = K.sum(tf.multiply((y_u - y_l), K_S)) / (K.sum(K_S) + 0.0001)
        PICP_S = K.mean(K_S)
        L_PICP = K.pow(K.relu((1. - alpha_) - PICP_S), 2) * 100  # PICP loss function
        # Calculate penalty function (Eq. 14 QD+ paper)
        Lp = K.mean((K.relu(y_l - y_o)) + (K.relu(y_o - y_u)))
        # Calculate loss (Eq. 12 QD+ paper)
        Loss_S = (1 - lambda_1) * (1 - lambda_2) * MPIW_c + lambda_1 * (
                1 - lambda_2) * L_PICP + lambda_2 * MSE + ksi * Lp
        return Loss_S

    #   @article{yu2020gradient,
    #   title={Gradient surgery for multi-task learning},
    #   author={Yu, Tianhe and Kumar, Saurabh and Gupta, Abhishek and Levine, Sergey and Hausman, Karol and Finn, Chelsea},
    #   journal={arXiv preprint arXiv:2001.06782},
    #   year={2020}}

    def compute_gradients_by_PCG(self, loss):
        assert type(loss) is list
        loss = tf.stack(loss)
        tf.random.shuffle(loss)
        # Compute per-task gradients.
        grads_task = tf.vectorized_map(lambda x: tf.concat([tf.reshape(grad, [-1, ])
                                                            for grad in tf.gradients(x, self.trainable_variables)
                                                            if grad is not None], axis=0), loss)
        num_tasks = loss.shape[0]

        # Compute gradient projections.
        def proj_grad(grad_task):
            for k in range(num_tasks):
                inner_product = tf.reduce_sum(grad_task * grads_task[k])
                proj_direction = inner_product / tf.reduce_sum(grads_task[k] * grads_task[k])
                grad_task = grad_task - tf.minimum(proj_direction, 0.) * grads_task[k]
            return grad_task

        proj_grads_flatten = tf.vectorized_map(proj_grad, grads_task)
        # Unpack flattened projected gradients back to their original shapes.
        proj_grads = []
        for j in range(num_tasks):
            start_idx = 0
            for idx, var in enumerate(self.trainable_variables):
                grad_shape = var.get_shape()
                flatten_dim = np.prod([grad_shape.dims[i].value for i in range(len(grad_shape.dims))])
                proj_grad = proj_grads_flatten[j][start_idx:start_idx + flatten_dim]
                proj_grad = tf.reshape(proj_grad, grad_shape)
                if len(proj_grads) < len(self.trainable_variables):
                    proj_grads.append(proj_grad)
                else:
                    proj_grads[idx] += proj_grad
                start_idx += flatten_dim

        grads_and_vars = zip(proj_grads, self.trainable_variables)
        return grads_and_vars

    def cagrad(self, loss, c=0.5):
        assert type(loss) is list
        loss = tf.stack(loss)
        tf.random.shuffle(loss)
        # Compute per-task gradients.
        grads = tf.vectorized_map(lambda x: tf.concat([tf.reshape(grad, [-1, ])
                                                       for grad in tf.gradients(x, self.trainable_variables)
                                                       if grad is not None], axis=0), loss)
        grads = tf.transpose(grads)
        g1 = grads[:, 0]
        g2 = grads[:, 1]
        g0 = (g1 + g2) / 2

        # g11 = g1.dot(g1).item()
        # g12 = g1.dot(g2).item()
        # g22 = g2.dot(g2).item()

        g11 = tf.reduce_sum(tf.multiply(g1, g1))
        g12 = tf.reduce_sum(tf.multiply(g1, g2))
        g22 = tf.reduce_sum(tf.multiply(g2, g2))

        # g0_norm = 0.5 * np.sqrt(g11 + g22 + 2 * g12 + 1e-4)
        g0_norm = 0.5 * tf.sqrt(g11 + g22 + 2 * g12 + 1e-4)
        # want to minimize g_w^Tg_0 + c*||g_0||*||g_w||
        coef = c * g0_norm

        def obj(x):
            # g_w^T g_0: x*0.5*(g11+g22-2g12)+(0.5+x)*(g12-g22)+g22
            # g_w^T g_w: x^2*(g11+g22-2g12)+2*x*(g12-g22)+g22
            return coef * tf.sqrt(x ** 2 * (g11 + g22 - 2 * g12) + 2 * x * (g12 - g22) + g22 + 1e-4) + \
                0.5 * x * (g11 + g22 - 2 * g12) + (0.5 + x) * (g12 - g22) + g22

        res = minimize_scalar(obj, bounds=(0, 1), method='bounded')
        x = res.x

        gw = x * g1 + (1 - x) * g2
        gw_norm = tf.sqrt(x ** 2 * g11 + (1 - x) ** 2 * g22 + 2 * x * (1 - x) * g12 + 1e-4)

        lmbda = coef / (gw_norm + 1e-4)
        g = g0 + lmbda * gw
        return g / (1 + c)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and on what you pass to `fit()`.
        X, y = data
        y_pred = self(X, training=True)  # Forward pass
        # cagrad_opt = CAGrad(self, learning_rate=0.001)
        # Compute the loss value (the loss function is configured in `compile()`)
        width_up_loss = self.selective_up(y, y_pred)
        width_low_loss = self.selective_low(y, y_pred)
        up_penalty_loss = self.up_penalty(y, y_pred)
        low_penalty_loss = self.low_penalty(y, y_pred)
        coverage_penalty = self.coverage_penalty(y, y_pred)
        abs_width_up_loss = self.abs_selective_up(y, y_pred)
        abs_width_low_loss = self.abs_selective_low(y, y_pred)
        qd_loss = self.qd_loss(y, y_pred)
        qd_plus_loss = self.qd_plus_loss(y, y_pred)
        mse_loss = self.mse_penalty(y, y_pred)  # 用于点估计
        C = self.adaptive_hyper(y, y_pred)  # 自适应lambda值
        dual_loss = self.dual_loss(y, y_pred, C)  # 2024年IEEE论文
        abs_ki_width_up = self.abs_ki_selective_up(y, y_pred)  # 加入ki的绝对值
        abs_ki_width_low = self.abs_ki_selective_low(y, y_pred)  # 加入ki的绝对值
        abs_mpiw = self.abs_mpiw(y, y_pred)  # 绝对值mpiw，原基础上合并
        picp_loss = self.picp_term(y, y_pred)  # picp正则项
        rmpiw_loss = self.abs_rmpiw_loss(y, y_pred)  # rmpiw
        dual_picp = self.dual_picp(y, y_pred)  # dual论文的PICP部分
        # Calculate the metrics
        coverage_value = self.coverage(y, y_pred)
        mpiw_value = self.mpiw(y, y_pred)

        trainable_vars = self.trainable_variables

        if self.method == 'normal':
            # loss = 1/4*K.mean(width_up_loss)+1/4*K.mean(width_low_loss)+1/4*coverage_penalty*K.mean(
            # up_penalty_loss)+1/4*coverage_penalty*K.mean(low_penalty_loss)
            """use abs width loss without ki and picp is from continue paper"""
            # loss = 1 / 5 * K.mean(abs_width_up_loss) + 1 / 5 * K.mean(abs_width_low_loss) + 1 / 5 * (
            #         coverage_penalty * K.mean(up_penalty_loss)) + 1 / 5 * (
            #                coverage_penalty * K.mean(low_penalty_loss))+1/5*mse_loss
            if self.loss_name == 'QD':
                loss = 1/2*qd_loss + 1/2*mse_loss
            elif self.loss_name == 'QD+':
                """use QD+ loss by snm paper, and it includes mse loss"""
                loss = qd_plus_loss
            elif self.loss_name == 'Continuous':
                """use Continuous paper without change"""
                loss = 1 / 3 * (K.mean(width_up_loss) + K.mean(width_low_loss)) \
                       + 1 / 3 * coverage_penalty * (K.mean(up_penalty_loss) + K.mean(low_penalty_loss)) \
                       + 1 / 3 * mse_loss
            elif self.loss_name == 'Dual':
                """use dual paper loss"""
                loss = dual_loss
            elif self.loss_name == 'MTLPI':
                loss = abs_mpiw \
                       + (1+C) * (K.mean(up_penalty_loss + low_penalty_loss))  \
                       + (1+C) * mse_loss
            elif self.loss_name == 'RMTLPI':
                loss = 1 / 3 * abs_mpiw \
                       + 1 / 3 * (K.mean(up_penalty_loss + low_penalty_loss)) * (1+C) \
                       + 1 / 3 * mse_loss
            """use abs width loss without ki and picp has adaptive hyper"""
            # loss = 1 / 5 * abs_width_up_loss + 1 / 5 * abs_width_low_loss + 1 / 5 * (
            #         (1+C) * K.mean(up_penalty_loss)) + 1 / 5 * (
            #                (1+C) * K.mean(low_penalty_loss)) + 1 / 5 * mse_loss
            """add ki and use abs width loss"""
            # loss = 1 / 5 * abs_ki_width_up + 1 / 5 * abs_ki_width_low + 1 / 5 * (
            #         (.5 + C) * K.mean(up_penalty_loss)) + 1 / 5 * (
            #                (.5 + C) * K.mean(low_penalty_loss)) + 1 / 5 * mse_loss
            """考虑到模型中已经存在mse，故删除，这里使用原论文"""
            # loss = 1 / 4 * K.mean(width_up_loss) + 1 / 4 * K.mean(width_low_loss) + 1 / 4 * (
            #         coverage_penalty * K.mean(up_penalty_loss)) + 1 / 4 * (
            #                coverage_penalty * K.mean(low_penalty_loss))+ 1 / 5 * mse_loss
            gradients = tf.gradients(loss, trainable_vars)

        elif self.method == 'PCGrad':
            loss = [K.mean(width_up_loss) + K.mean(width_low_loss),
                    (K.mean(up_penalty_loss) + K.mean(low_penalty_loss)) * K.mean(coverage_penalty) + K.epsilon(),
                    mse_loss + K.epsilon()]
            computegrad = self.compute_gradients_by_PCG(loss)

        # Update weights
        if self.method == 'normal':
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        elif self.method == 'PCGrad':
            self.optimizer.apply_gradients(computegrad)

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
