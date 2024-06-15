import tensorflow as tf
import numpy as np
from keras import backend as K
from scipy.optimize import minimize, minimize_scalar


class MultiTaskModel(tf.keras.Model):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        # 定义共享层
        self.shared_layer = tf.keras.layers.Dense(128, activation='relu')
        # 定义任务特定层
        self.task1_layer = tf.keras.layers.Dense(64, activation='relu')
        self.task1_output = tf.keras.layers.Dense(1, name='task1_output')

        self.task2_layer = tf.keras.layers.Dense(64, activation='relu')
        self.task2_output = tf.keras.layers.Dense(1, name='task2_output')

    def call(self, inputs):
        shared_rep = self.shared_layer(inputs)
        task1_rep = self.task1_layer(shared_rep)
        task1_output = self.task1_output(task1_rep)

        task2_rep = self.task2_layer(shared_rep)
        task2_output = self.task2_output(task2_rep)

        return task1_output, task2_output


class CAGrad:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def compute_loss_and_grads(self, x, y1, y2):
        with tf.GradientTape(persistent=True) as tape:
            y1_pred, y2_pred = self.model(x)
            loss1 = tf.reduce_mean(tf.keras.losses.mean_squared_error(y1, y1_pred))
            loss2 = tf.reduce_mean(tf.keras.losses.mean_squared_error(y2, y2_pred))

        grads_task1 = tape.gradient(loss1, self.model.trainable_variables)
        grads_task2 = tape.gradient(loss2, self.model.trainable_variables)

        # Replace None gradients with zero gradients
        grads_task1 = [tf.zeros_like(var) if grad is None else grad for var, grad in
                       zip(self.model.trainable_variables, grads_task1)]
        grads_task2 = [tf.zeros_like(var) if grad is None else grad for var, grad in
                       zip(self.model.trainable_variables, grads_task2)]

        return loss1, loss2, grads_task1, grads_task2

    def solve_weights(self, grads_task1, grads_task2):
        # 将梯度转换为向量形式
        g1 = tf.concat([tf.reshape(g, [-1]) for g in grads_task1], axis=0)
        g2 = tf.concat([tf.reshape(g, [-1]) for g in grads_task2], axis=0)

        # 求解二次规划问题
        G = tf.stack([g1, g2], axis=1)
        GtG = tf.matmul(G, G, transpose_a=True)
        q = -tf.linalg.diag_part(GtG)
        A = tf.ones((1, 2))
        b = tf.constant([1.0])

        # 使用 TensorFlow 求解二次规划
        P = tf.linalg.inv(GtG)
        q = tf.expand_dims(q, axis=-1)
        A = tf.expand_dims(A, axis=0)
        b = tf.expand_dims(b, axis=0)

        weights = tf.linalg.solve(P, q)
        weights = tf.nn.softmax(weights)  # 确保权重非负并且和为1

        return tf.reshape(weights, [-1]).numpy()

    def optimize(self, grads_task1, grads_task2, weights):
        grads_combined = []
        for g1, g2 in zip(grads_task1, grads_task2):
            g_combined = weights[0] * g1 + weights[1] * g2
            grads_combined.append(g_combined)

        self.optimizer.apply_gradients(zip(grads_combined, self.model.trainable_variables))


def cagrad(grads, num_tasks, alpha=0.5, rescale=1):
    GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
    g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

    x_start = np.ones(num_tasks) / num_tasks
    bnds = tuple((0, 1) for x in x_start)
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha * g0_norm + 1e-8).item()

    def objfn(x):
        return (x.reshape(1, num_tasks).dot(A).dot(b.reshape(num_tasks, 1)) + c * np.sqrt(
            x.reshape(1, num_tasks).dot(A).dot(x.reshape(num_tasks, 1)) + 1e-8)).sum()

    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    w_cpu = res.x
    ww = tf.Tensor(w_cpu).to(grads.device)
    gw = (grads * ww.view(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm + 1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale == 0:
        return g
    elif rescale == 1:
        return g / (1 + alpha ** 2)
    else:
        return g / (1 + alpha)


def cagrad2(grads, c=0.5):
    g1 = grads[:,0]
    g2 = grads[:,1]
    g0 = (g1+g2)/2

    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()

    g0_norm = 0.5 * np.sqrt(g11+g22+2*g12+1e-4)

    # want to minimize g_w^Tg_0 + c*||g_0||*||g_w||
    coef = c * g0_norm

    def obj(x):
        # g_w^T g_0: x*0.5*(g11+g22-2g12)+(0.5+x)*(g12-g22)+g22
        # g_w^T g_w: x^2*(g11+g22-2g12)+2*x*(g12-g22)+g22
        return coef * np.sqrt(x**2*(g11+g22-2*g12)+2*x*(g12-g22)+g22+1e-4) + \
                0.5*x*(g11+g22-2*g12)+(0.5+x)*(g12-g22)+g22

    res = minimize_scalar(obj, bounds=(0,1), method='bounded')
    x = res.x

    gw = x * g1 + (1-x) * g2
    gw_norm = np.sqrt(x**2*g11+(1-x)**2*g22+2*x*(1-x)*g12+1e-4)

    lmbda = coef / (gw_norm+1e-4)
    g = g0 + lmbda * gw
    return g / (1+c)


if __name__ == '__main__':
    # 生成一些示例数据
    np.random.seed(42)
    x_data = np.random.rand(1000, 10).astype(np.float32)
    y1_data = np.random.rand(1000, 1).astype(np.float32)
    y2_data = np.random.rand(1000, 1).astype(np.float32)

    # 定义模型和优化器
    model = MultiTaskModel()
    cagrad_optimizer = CAGrad(model)

    # 训练模型
    epochs = 50
    batch_size = 32

    dataset = tf.data.Dataset.from_tensor_slices((x_data, y1_data, y2_data)).batch(batch_size)

    for epoch in range(epochs):
        for step, (x_batch, y1_batch, y2_batch) in enumerate(dataset):
            loss1, loss2, grads_task1, grads_task2 = cagrad_optimizer.compute_loss_and_grads(x_batch, y1_batch, y2_batch)
            weights = cagrad_optimizer.solve_weights(grads_task1, grads_task2)
            cagrad_optimizer.optimize(grads_task1, grads_task2, weights)

        print(f'Epoch {epoch + 1}, Loss1: {loss1.numpy()}, Loss2: {loss2.numpy()}')
