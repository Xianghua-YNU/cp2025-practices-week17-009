"""
学生模板：松弛迭代法解常微分方程
文件：relaxation_method_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt

def solve_ode(h, g, max_iter=10000, tol=1e-6):
    """
    实现松弛迭代法求解常微分方程 d²x/dt² = -g
    边界条件：x(0) = x(10) = 0（抛体运动问题）
    
    参数:
        h (float): 时间步长
        g (float): 重力加速度
        max_iter (int): 最大迭代次数
        tol (float): 收敛容差
    
    返回:
        tuple: (时间数组, 解数组)
    
    物理背景: 质量为1kg的球从高度x=0抛出，10秒后回到x=0
    数值方法: 松弛迭代法，迭代公式 x(t) = 0.5*h²*g + 0.5*[x(t+h)+x(t-h)]
    
    实现步骤:
    1. 初始化时间数组和解数组
    2. 应用松弛迭代公式直到收敛
    3. 返回时间和解数组
    """
    # 初始化时间数组 [0, 10] 区间，步长为 h
    t = np.arange(0, 10 + h, h)
    
    # 初始化解数组，边界条件已满足：x[0] = x[-1] = 0
    x = np.zeros(t.size)
    
    # TODO: 实现松弛迭代算法
    # 提示：
    # 1. 设置初始变化量 delta = 1.0
    # 2. 当 delta > tol 时继续迭代
    # 3. 对内部点应用公式：x_new[1:-1] = 0.5 * (h*h*g + x[2:] + x[:-2])
    # 4. 计算最大变化量：delta = np.max(np.abs(x_new - x))
    # 5. 更新解：x = x_new
    
    # 初始化时间数组 [0, 10] 区间，步长为 h
    t = np.arange(0, 10 + h, h)
    
    # 初始化解数组，初始值为全零（满足边界条件）
    x = np.zeros(t.size)
    
    # 初始化收敛判断变量和迭代计数器
    delta = 1.0        # 记录每次迭代的最大变化量
    iteration = 0      # 当前迭代次数
    
    while delta > tol and iteration < max_iter:    # 未达到收敛且未超过最大迭代次数时继续
        x_new = np.copy(x)
        
        # 核心松弛法更新公式：
        # 使用中心差分法离散二阶导数 d²x/dt² ≈ (x[i+1] - 2x[i] + x[i-1])/h²
        # 代入方程 (x[i+1] - 2x[i] + x[i-1])/h² = -g
        # 整理得：x[i] = (h²g + x[i+1] + x[i-1])/2
        x_new[1:-1] = 0.5 * (h * h * g + x[2:] + x[:-2])
        
        # 计算当前迭代的最大变化量（无穷范数）
        delta = np.max(np.abs(x_new - x))
        
        # 更新解为当前迭代结果
        x = x_new
        iteration += 1
    
    return t, x

if __name__ == "__main__":
    # 测试参数
    h = 10 / 100  # 时间步长
    g = 9.8       # 重力加速度
    
    # 调用求解函数
    t, x = solve_ode(h, g)
    
    # 绘制结果
    plt.plot(t, x)
    plt.xlabel('时间 (s)')
    plt.ylabel('高度 (m)')
    plt.title('抛体运动轨迹 (松弛迭代法)')
    plt.grid()
    plt.show()
