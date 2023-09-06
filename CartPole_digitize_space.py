import gym
import numpy as np

# state discretization
NUM_DIGITIZED = 6


# 分桶
def bins(clip_min, clip_max, nim_bins=NUM_DIGITIZED):
    return np.linspace(clip_min, clip_max, nim_bins + 1)[1: -1]  # 创建一个包含num_bins + 1个元素的等差数列并排除第一个和最后一个元素（即最小值和最大值）


# env.observation_space: Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32)
# 将四位6进制数映射为id （数字乘以NUM_DIGITIZED ** n_states的加和）
def digitize_state(observation, NUM_DIGITIZED):
    pos, cart_v, angle, pole_v = observation
    digitized = [np.digitize(pos, bins=bins(-4.8, 4.8, NUM_DIGITIZED)),
                 np.digitize(cart_v, bins=bins(-3., 3, NUM_DIGITIZED)),
                 np.digitize(angle, bins=bins(-0.418, 0.418, NUM_DIGITIZED)),
                 np.digitize(pole_v, bins=bins(-2, 2, NUM_DIGITIZED))]

    ind = sum(
        [d * (NUM_DIGITIZED ** i) for i, d in enumerate(digitized)])  # 将数字列表 digitized 中的各个数字按权重加权并相加，得到整数 ind 的十进制表示
    return ind

######## example ########
# bins(-2.4, 2.4, NUM_DIGITIZED)  # array([-1.6, -0.8, 0., 0.8, 1.6])
