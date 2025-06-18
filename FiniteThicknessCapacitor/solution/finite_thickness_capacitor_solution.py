#!/usr/bin/env python3
"""
Module: Finite Thickness Parallel Plate Capacitor Solution
File: finite_thickness_capacitor_solution.py

Solves the Laplace equation for finite thickness parallel plate capacitor
using Gauss-Seidel SOR method and calculates charge density distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from scipy.ndimage import laplace

def solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega=1.9, max_iter=10000, tolerance=1e-6):
    """
    使用逐次超松弛(SOR)方法求解二维拉普拉斯方程
    
    参数：
        nx (int): x方向的网格点数
        ny (int): y方向的网格点数  
        plate_thickness (int): 导体板厚度(网格点数)
        plate_separation (int): 板间距离(网格点数)
        omega (float): 松弛因子(1.0 < omega < 2.0)
        max_iter (int): 最大迭代次数
        tolerance (float): 收敛容差
        
    返回：
        np.ndarray: 二维电势分布数组
    """
    # 初始化电势网格
    U = np.zeros((ny, nx))
    
    # 创建导体区域掩模
    conductor_mask = np.zeros((ny, nx), dtype=bool)
    
    # 定义导体区域
    # 上极板：+100V
    conductor_left = nx//4
    conductor_right = nx//4*3
    y_upper_start = ny // 2 + plate_separation // 2
    y_upper_end = y_upper_start + plate_thickness
    conductor_mask[y_upper_start:y_upper_end, conductor_left:conductor_right] = True
    U[y_upper_start:y_upper_end, conductor_left:conductor_right] = 100.0
    
    # 下极板：-100V
    y_lower_end = ny // 2 - plate_separation // 2
    y_lower_start = y_lower_end - plate_thickness
    conductor_mask[y_lower_start:y_lower_end, conductor_left:conductor_right] = True
    U[y_lower_start:y_lower_end, conductor_left:conductor_right] = -100.0
    
    # 边界条件：接地边界
    U[:, 0] = 0.0
    U[:, -1] = 0.0
    U[0, :] = 0.0
    U[-1, :] = 0.0
    
    # SOR迭代
    for iteration in range(max_iter):
        U_old = U.copy()
        max_error = 0.0
        
        # 更新内部点(排除导体和边界)
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                if not conductor_mask[i, j]:  # 跳过导体点
                    # SOR更新公式
                    U_new = 0.25 * (U[i+1, j] + U[i-1, j] + U[i, j+1] + U[i, j-1])
                    U[i, j] = (1 - omega) * U[i, j] + omega * U_new
                    
                    # 跟踪最大误差
                    error = abs(U[i, j] - U_old[i, j])
                    max_error = max(max_error, error)
        
        # 检查收敛
        if max_error < tolerance:
            print(f"在 {iteration + 1} 次迭代后收敛")
            break
    else:
        print(f"警告：达到最大迭代次数 ({max_iter})")
    
    return U

def calculate_charge_density(potential_grid, dx, dy):
    """
    使用泊松方程计算电荷密度：rho = -1/(4*pi) * nabla^2(U)
    
    参数：
        potential_grid (np.ndarray): 二维电势分布
        dx (float): x方向网格间距
        dy (float): y方向网格间距
        
    返回：
        np.ndarray: 二维电荷密度分布
    """
    # 使用scipy.ndimage.laplace计算拉普拉斯算子
    laplacian_U = laplace(potential_grid, mode='nearest') / (dx**2)  # 假设dx=dy
    
    # 根据泊松方程计算电荷密度
    rho = -laplacian_U / (4 * np.pi)
    
    return rho

def plot_results(potential, charge_density, x_coords, y_coords):
    """
    创建结果的可视化图形（使用最初代码的颜色方案）
    
    参数：
        potential (np.ndarray): 二维电势分布
        charge_density (np.ndarray): 电荷密度分布
        x_coords (np.ndarray): x坐标数组
        y_coords (np.ndarray): y坐标数组
    """
    plt.figure(figsize=(15, 6))
    
    # 子图1：电势分布等高线图（使用viridis颜色图）
    plt.subplot(1, 2, 1)
    contour = plt.contourf(x_coords, y_coords, potential, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Potential (V)')
    plt.title('Electric Potential Distribution')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    
    # 子图2：电荷密度分布（使用plasma颜色图）
    plt.subplot(1, 2, 2)
    charge_contour = plt.contourf(x_coords, y_coords, charge_density, levels=20, cmap='plasma')
    plt.colorbar(charge_contour, label='Charge Density (C/m²)')
    plt.title('Charge Density Distribution')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 仿真参数设置
    nx, ny = 100, 100  # 网格尺寸
    plate_thickness = 10  # 导体板厚度(网格点数)
    plate_separation = 30  # 板间距离(网格点数)
    omega = 1.9  # SOR松弛因子
    
    # 物理尺寸设置
    Lx, Ly = 1.0, 1.0  # 计算域尺寸(米)
    dx = Lx / (nx - 1)  # x方向网格间距
    dy = Ly / (ny - 1)  # y方向网格间距
    
    # 创建坐标数组
    x_coords = np.linspace(0, Lx, nx)
    y_coords = np.linspace(0, Ly, ny)
    
    # 打印仿真信息
    print("正在求解有限厚度平行板电容器...")
    print(f"网格尺寸: {nx} x {ny}")
    print(f"极板厚度: {plate_thickness} 个网格点")
    print(f"极板间距: {plate_separation} 个网格点")
    print(f"SOR松弛因子: {omega}")
    
    # 求解拉普拉斯方程
    start_time = time.time()
    potential = solve_laplace_sor(
        nx, ny, plate_thickness, plate_separation, omega
    )
    solve_time = time.time() - start_time
    
    print(f"求解完成，耗时 {solve_time:.2f} 秒")
    
    # 计算电荷密度
    charge_density = calculate_charge_density(potential, dx, dy)
    
    # 可视化结果（使用最初代码的颜色方案）
    plot_results(potential, charge_density, x_coords, y_coords)
    
    # 打印统计信息
    print(f"\n电势统计:")
    print(f"  最小电势: {np.min(potential):.2f} V")
    print(f"  最大电势: {np.max(potential):.2f} V")
    print(f"  电势范围: {np.max(potential) - np.min(potential):.2f} V")
    
    print(f"\n电荷密度统计:")
    print(f"  最大电荷密度: {np.max(np.abs(charge_density)):.6f} C/m²")
    print(f"  总正电荷: {np.sum(charge_density[charge_density > 0]) * dx * dy:.6f} C")
    print(f"  总负电荷: {np.sum(charge_density[charge_density < 0]) * dx * dy:.6f} C")
    print(f"  总电荷: {np.sum(charge_density) * dx * dy:.6f} C")
