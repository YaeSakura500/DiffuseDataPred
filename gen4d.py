import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import h5py

def h5_data_read(filename):
    file = h5py.File(filename, 'r', )
    u = file['u'][:].astype('float32')
    v = file['v'][:].astype('float32')
    w = file['w'][:].astype('float32')
    f1 = file['f1'][:].astype('float32')
    f2 = file['f2'][:].astype('float32')
    f3 = file['f3'][:].astype('float32')

    return u, v, w, f1, f2, f3

# a = h5_data_read('data_diffusion/diffusion_data_0.h5')
# print('a')

def velocity_generate(mu, l, size_s, size_t, lengths_s, lengths_t):
    def fftfreq_4d(shape, unit_lengths):
        """生成四维的波数网格"""
        all_k = [np.fft.fftfreq(s, d=d) for s, d in zip(shape, unit_lengths)]
        return np.meshgrid(*all_k, indexing="ij")

    def generate_field_4d(statistic, power_spectrum, shape, unit_length, delta, stat_real=False):
        """
        Generates a 4D field (space-time field) given a statistic and a power spectrum.

        Parameters
        ----------
        statistic : callable
            A function that generates random values in Fourier space.

        power_spectrum : callable
            A function that returns the power spectrum at a given wave number.

        shape : tuple
            The shape of the output field, e.g., (nx, ny, nz, nt) for a 3D spatial field + time.

        unit_length : float, optional
            The physical length per pixel, controls the wave number scale.

        stat_real : bool, optional
            Whether to generate the distribution in real space or Fourier space.

        Returns
        -------
        field : ndarray
            A 4D real-valued field with the given power spectrum.
        """
        # Generate 4D frequency grid (space + time)
        kgrid = fftfreq_4d(shape, unit_length)
        knorm = np.sqrt(np.sum(np.power(kgrid, 2), axis=0))  # 计算波数模长

        # Generate random field in Fourier space
        if stat_real:
            field = statistic(shape)  # 在真实空间生成随机场
            fftfield = np.fft.fftn(field)  # 对场进行傅里叶变换
        else:
            fftfield = statistic(knorm.shape)  # 在傅里叶空间生成随机场

        # Apply power spectrum
        power_k = np.zeros_like(knorm)
        mask = knorm > 0
        power_k[mask] = 1e10 * np.sqrt(power_spectrum(knorm[mask]) * delta[0, 0] * delta[0, 1] * delta[0, 2] * delta[
            0, 3])  # Apply power spectrum to the field
        # power_k[mask] = np.sqrt(power_spectrum(knorm[mask]))
        fftfield *= power_k

        # Perform inverse FFT to get back to real space
        return np.real(np.fft.ifftn(fftfield))  # 转换回真实空间

    def matern_psd(k, sigma2=1.0, mu=0.5, l=1.0, n=4):  # 这样就只是k的函数
        """Calculate the Matern power spectrum."""
        factor1 = (sigma2 ** 2) * (2 ** n) * (np.pi ** (n / 2))
        factor2 = gamma(mu + n / 2) / (gamma(mu) * l ** (2 * mu))
        spectrum = factor1 * factor2 * (2 * mu / l ** 2 + (2 * np.pi * k) ** 2) ** (-(mu + n / 2)) * ((2 * mu) ** mu)
        return spectrum

    def generate_matern_field_4d(shape, mu, l, unit_length, delta, sigma2=1.0):
        """Generate a 4D (space-time) Matern field."""

        def power_spectrum(k):
            return matern_psd(k, sigma2=sigma2, mu=mu, l=l)

        def distrib(shape):
            # Generating random complex Gaussian field (Fourier space)
            a = np.random.normal(0, 1, size=shape)
            b = np.random.normal(0, 1, size=shape)
            return a + 1j * b

        # Generate the 4D field
        field = generate_field_4d(distrib, power_spectrum, shape, unit_length, delta)
        return field

    # Parameters
    shape = (size_s, size_s, size_s, size_t)  # (nx, ny, nz, nt), 64x64x64 spatial grid and 100 time steps
    # 每个维度的采样间隔
    unit_lengths = (lengths_s, lengths_s, lengths_s, lengths_t)
    # 计算每个方向的分辨率 Δk = 1 / (N * d)
    resolutions = np.array([1 / (N * d) for N, d in zip(shape, unit_lengths)])
    # 调整成 (1, 4) 的形状
    delta = resolutions.reshape(1, 4)
    # mu = 3  # Smoothness parameter for the Matern covariance
    # l = 100  # Spatial correlation length 1 10 100
    sigma2 = 1.0  # Variance
    # Generate 4D random field
    field1 = generate_matern_field_4d(shape, mu, l, unit_lengths, delta, sigma2)
    field2 = generate_matern_field_4d(shape, mu, l, unit_lengths, delta, sigma2)
    field3 = generate_matern_field_4d(shape, mu, l, unit_lengths, delta, sigma2)

    d_field1 = np.gradient(field1)
    d_field2 = np.gradient(field2)
    d_field3 = np.gradient(field3)

    u_field = d_field3[1] - d_field2[2]
    v_field = -(d_field3[0] - d_field1[2])
    w_field = d_field2[0] - d_field1[1]

    divergence = np.gradient(u_field, 1, axis=0) + \
                 np.gradient(v_field, 1, axis=1) + \
                 np.gradient(w_field, 1, axis=2)
    print(f"Max divergence in the flow field: {np.max(np.abs(divergence))}")
    return field1, field2, field3, u_field, v_field, w_field

def diffusion_eq(field, gama):
    d_field = np.gradient(field)
    dxx = np.gradient(d_field[0])[0]
    dyy = np.gradient(d_field[1])[1]
    dzz = np.gradient(d_field[2])[2]
    soruce = d_field[3] - gama*(dxx + dyy + dzz)
    return soruce
def data_generate(num):
    def diffusion_eq(field, gama):
        d_field = np.gradient(field)
        dxx = np.gradient(d_field[0])[0]
        dyy = np.gradient(d_field[1])[1]
        dzz = np.gradient(d_field[2])[2]
        soruce = d_field[3] - gama*(dxx + dyy + dzz)
        return soruce

    mulist = [1, 2, 3]
    gamalist = [0.1, 1e-3, 1e-5]
    for i in range(num):
        for mu in mulist:
            for gama in gamalist:
                _, _, _, u, v, w = velocity_generate(mu, 100, size_s=64, size_t=64, lengths_s=1, lengths_t=1)
                f1 = diffusion_eq(u, gama=gama)
                f2 = diffusion_eq(v, gama=gama)
                f3 = diffusion_eq(w, gama=gama)

                with h5py.File('data_diffusion/diffusion_data_mu{0}_ga{1}_{2}.h5'.format(
                    mu, gama, i
                ), 'w') as f:
                    f.create_dataset('u', data=u.astype('float16'))
                    f.create_dataset('v', data=v.astype('float16'))
                    f.create_dataset('w', data=w.astype('float16'))
                    f.create_dataset('f1', data=f1.astype('float16'))
                    f.create_dataset('f2', data=f2.astype('float16'))
                    f.create_dataset('f3', data=f3.astype('float16'))

data_generate(1)


# _, _, _, u_field, v_field, w_field = velocity_generate(3, 100, size_s=64, size_t=64, lengths_s=1, lengths_t=1)
#
# u_field = diffusion_eq(u_field, 1e-5)

def period_check(field, max_s=64, max_t=64):
    fig, axes = plt.subplots(2, 4, figsize=(6, 6))
    axes[0, 0].imshow(field[:, :, 0, 50], cmap="viridis", aspect="auto")
    axes[0, 0].set_title(f"z=0", fontsize=20)
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[1, 0].imshow(field[:, :, max_s, 50], cmap="viridis", aspect="auto")
    axes[1, 0].set_title(f"z=99", fontsize=20)
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")
    axes[0, 1].imshow(field[:, 0, :, 50], cmap="viridis", aspect="auto")
    axes[0, 1].set_title(f"y=0", fontsize=20)
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("z")
    axes[1, 1].imshow(field[:, max_s, :, 50], cmap="viridis", aspect="auto")
    axes[1, 1].set_title(f"y=99", fontsize=20)
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("z")
    axes[0, 2].imshow(field[0, :, :, 50], cmap="viridis", aspect="auto")
    axes[0, 2].set_title(f"x=0", fontsize=20)
    axes[0, 2].set_xlabel("y")
    axes[0, 2].set_ylabel("z")
    axes[1, 2].imshow(field[max_s, :, :, 50], cmap="viridis", aspect="auto")
    axes[1, 2].set_title(f"x=99", fontsize=20)
    axes[1, 2].set_xlabel("y")
    axes[1, 2].set_ylabel("z")
    axes[0, 3].imshow(field[:, :, 50, 0], cmap="viridis", aspect="auto")
    axes[0, 3].set_title(f"t=0", fontsize=20)
    axes[0, 3].set_xlabel("x")
    axes[0, 3].set_ylabel("y")
    axes[1, 3].imshow(field[:, :, 50, max_t], cmap="viridis", aspect="auto")
    axes[1, 3].set_title(f"t=99", fontsize=20)
    axes[1, 3].set_xlabel("x")
    axes[1, 3].set_ylabel("y")
    plt.tight_layout()
    plt.show()


# period_check(u_field, 63, 63)


def Spatial_Slice_at_Time(field):
    time_index1 = 50  # 选择一个时间步
    z_index3 = 33  # 选择一个空间切片（z维度）
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))

    # 绘制第一个时间步的图像
    im1 = axes.imshow(field[:, :, z_index3, time_index1], cmap="viridis", aspect="auto")
    axes.set_title(f"2D Spatial Slice at Time {time_index1} and z = {z_index3}", fontsize=20)
    axes.set_xlabel("x")
    axes.set_ylabel("y")

    # 添加共享颜色条
    fig.colorbar(im1, orientation='vertical', label="Field Value")

    # 显示图像
    plt.tight_layout()
    plt.show()


def dx_dy_dz_dt(field):
    dx = np.gradient(field, axis=0)  # x 方向导数
    dy = np.gradient(field, axis=1)  # y 方向导数
    dz = np.gradient(field, axis=2)  # z 方向导数
    dt = np.gradient(field, axis=3)  # 时间方向导数

    z_index = 30  # 固定 z 切片
    t_index = 50  # 固定时间步

    plt.figure(figsize=(24, 6))
    plt.subplot(1, 4, 1)
    plt.title("dx at z=30, t=50", fontsize=24)
    plt.imshow(dx[:, :, z_index, t_index], cmap="coolwarm")
    plt.colorbar(orientation='vertical', )

    plt.subplot(1, 4, 2)
    plt.title("dy at z=30, t=50", fontsize=24)
    plt.imshow(dy[:, :, z_index, t_index], cmap="coolwarm")
    plt.colorbar(orientation='vertical', )

    plt.subplot(1, 4, 3)
    plt.title("dz at z=30, t=50", fontsize=24)
    plt.imshow(dz[:, :, z_index, t_index], cmap="coolwarm")
    plt.colorbar(orientation='vertical', )

    plt.subplot(1, 4, 4)
    plt.title("dt at z=30, t=50", fontsize=24)
    plt.imshow(dt[:, :, z_index, t_index], cmap="coolwarm")
    plt.colorbar(orientation='vertical', )

    plt.tight_layout()
    plt.show()


# Spatial_Slice_at_Time(u_field)
# dx_dy_dz_dt(u_field)

#
# # 创建一个包含3个子图的图形
# fig, axes = plt.subplots(1, 3, figsize=(20, 6))
#
# # 绘制第一个时间步的图像
# im1 = axes[0].imshow(field[:, :, z_index1, time_index1], cmap="viridis", aspect="auto")
# axes[0].set_title(f"2D Spatial Slice at Time {time_index1} and z = {z_index1}")
# axes[0].set_xlabel("x")
# axes[0].set_ylabel("y")
#
# # 绘制第二个时间步的图像
# im2 = axes[1].imshow(field[:, :, z_index1, time_index2], cmap="viridis", aspect="auto")
# axes[1].set_title(f"2D Spatial Slice at Time {time_index2} and z = {z_index1}")
# axes[1].set_xlabel("x")
#
# # 绘制第三个时间步的图像
# im3 = axes[2].imshow(field[:, :, z_index1, time_index3], cmap="viridis", aspect="auto")
# axes[2].set_title(f"2D Spatial Slice at Time {time_index3} and z = {z_index1}")
# axes[2].set_xlabel("x")
#
# # 添加共享颜色条
# fig.colorbar(im1, orientation='vertical', label="Field Value")
#
# # 显示图像
# plt.tight_layout()
# # plt.show()
#
# # 创建一个包含3个子图的图形
# fig, axes = plt.subplots(1, 3, figsize=(20, 6))
#
# # 绘制第一个时间步的图像
# im1 = axes[0].imshow(field[:, :, z_index2, time_index1], cmap="viridis", aspect="auto")
# axes[0].set_title(f"2D Spatial Slice at Time {time_index1} and z = {z_index2}")
# axes[0].set_xlabel("x")
# axes[0].set_ylabel("y")
#
# # 绘制第二个时间步的图像
# im2 = axes[1].imshow(field[:, :, z_index2, time_index2], cmap="viridis", aspect="auto")
# axes[1].set_title(f"2D Spatial Slice at Time {time_index2} and z = {z_index2}")
# axes[1].set_xlabel("x")
#
# # 绘制第三个时间步的图像
# im3 = axes[2].imshow(field[:, :, z_index2, time_index3], cmap="viridis", aspect="auto")
# axes[2].set_title(f"2D Spatial Slice at Time {time_index3} and z = {z_index2}")
# axes[2].set_xlabel("x")
#
# # 添加共享颜色条
# fig.colorbar(im1, orientation='vertical', label="Field Value")
#
# # 显示图像
# plt.tight_layout()
# # plt.show()
#
# 创建一个包含3个子图的图形

#
#
#
# # 假设 z_index 和 time_indices 是你要选择的切片和时间步索引
# z_indices = [10, 20, 30]  # 假设我们有 3 个不同的 z 切片
# time_indices = [50, 60, 70]  # 5 个不同的时间步
#
# # 创建一个包含 3x3 子图的图形
# fig, axes = plt.subplots(3, 3, figsize=(20, 15), constrained_layout=True)
#
# # 变量来追踪当前的子图位置
# index = 0
#
# # 遍历每个子图并绘制相应的图像
# for i in range(3):
#     for j in range(3):
#         # 获取当前的 z 切片和时间步索引
#         z_index = z_indices[i]  # 选择对应的 z 切片
#         time_index = time_indices[j]  # 选择对应的时间步
#
#         # 绘制当前切片和时间步的图像
#         im = axes[i, j].imshow(field[:, :, z_index, time_index], cmap="viridis", aspect="auto")
#         axes[i, j].set_title(f"Time {time_index}, z={z_index}")
#         axes[i, j].set_xlabel("x")
#         axes[i, j].set_ylabel("y" if j == 0 else "")  # 只有第一列显示 y 标签
#
# # 为所有子图添加一个共享颜色条
# fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
# im.set_label("Field Value")
#
# # 显示图像
# # plt.show()
#
#
# 计算各方向的一阶导数


#
# # 假设 field 是形状为 (m, n, p, t) 的高斯场
# # field[x, y, z, t] 表示 (x, y, z) 点在时间 t 的高斯场值
#
# # # 计算空间和时间的导数
# dx_field = np.gradient(field, axis=0)  # x方向的导数
# dy_field = np.gradient(field, axis=1)  # y方向的导数
# dz_field = np.gradient(field, axis=2)  # z方向的导数
# dt_field = np.gradient(field, axis=3)  # 时间方向的导数
#
# # 查看某个点的导数，假设选定 z=30 和 t=50
# z_index = 30
# time_index = 50
#
# # 计算空间梯度的幅度
# gradient_magnitude = np.sqrt(dx_field[:, :, z_index, time_index]**2 +
#                              dy_field[:, :, z_index, time_index]**2 +
#                              dz_field[:, :, z_index, time_index]**2)
#
# # 绘制空间梯度的幅度
# plt.imshow(gradient_magnitude, cmap="coolwarm", aspect="auto")
# plt.colorbar(label="Gradient Magnitude")
# plt.title(f"Gradient Magnitude at z={z_index}, Time {time_index}")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()
#
#
# # 计算时间方向的导数
# time_derivative = np.abs(dt_field[:, :, z_index, time_index])
#
# # 绘制时间导数的幅度
# plt.imshow(time_derivative, cmap="coolwarm", aspect="auto")
# plt.colorbar(label="Time Derivative Magnitude")
# plt.title(f"Time Derivative at z={z_index}, Time {time_index}")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()
#
# # # 计算导数的统计特性
# # mean_dx = np.mean(dx_field)
# # std_dx = np.std(dx_field)
# # max_dx = np.max(dx_field)
# # min_dx = np.min(dx_field)
# #
# # # 输出统计结果
# # print(f"Mean: {mean_dx:.3f}, Std: {std_dx:.3f}, Max: {max_dx:.3f}, Min: {min_dx:.3f}")
# #
# # # 判断标准：标准差过大或极值差过大，可能存在不连续性
# # if std_dx > threshold_std or (max_dx - min_dx) > threshold_range:
# #     print("Potential discontinuity or non-smooth region detected.")
# # else:
# #     print("The gradient appears smooth.")
#
# # 计算四分位数和异常点阈值
# q1, q3 = np.percentile(dx_field, [25, 75])
# iqr = q3 - q1
# lower_bound = q1 - 1.5 * iqr
# upper_bound = q3 + 1.5 * iqr
#
# # 标记异常点
# outliers = (dx_field < lower_bound) | (dx_field > upper_bound)
#
# # 可视化异常点
# plt.imshow(outliers[:, :, z_index, time_index], cmap="gray", aspect="auto")
# plt.title("Outlier Map (Potential Discontinuities)")
# plt.xlabel("x")
# plt.ylabel("y")
# # plt.show()
