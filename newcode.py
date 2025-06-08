import numpy as np
from itertools import product
import random

# 设置随机种子
# np.random.seed(42)
# random.seed(42)


class UAV:
    """
    无人机类 - 负责UAV飞行、计算任务生成、通信
    """

    def __init__(self, uav_id, area_size=(1000, 1000, 100), is_debug=False):
        """
        初始化单个UAV
        :param uav_id: UAV编号
        :param area_size: 飞行区域大小
        :param is_debug: 调试模式
        """
        self.uav_id = uav_id
        self.area_size = area_size
        self.is_debug = is_debug

        # 状态变量（将在系统初始化时设置）
        self.positions = None  # 位置轨迹
        self.velocities = None  # 速度轨迹
        self.transmission_power = None  # 发射功率

        # UAV飞行参数 (方程1-3)
        self.v_max = 50  # 最大速度 m/s
        self.a_max = 5  # 最大加速度 m/s^2
        self.delta_t = 1  # 时隙长度 s

        # 通信参数 (方程4)
        self.B = 20e6  # 上行带宽 Hz
        self.sigma_squared = 1e-10  # 噪声功率
        self.num_antennas = 1  # 天线数量

        # 无人机发射功率参数
        self.P_min = 0.01
        self.P_max = 1.0
        self.P_levels = 5
        # 生成指定范围内均匀间隔的数值序列
        self.power_options = np.linspace(self.P_min, self.P_max, self.P_levels)

        # 本地计算能力参数 (方程14-16)
        self.f_local = 5e9  # 本地CPU处理能力 Hz
        self.lambda_local = 1e-28  # 本地执行能耗系数

        # 反射系数（用于感知模型，方程7）
        self.reflection_coefficient = random.uniform(0.1, 1.0)

        # 计算任务（UAV生成的）
        self.computation_tasks = {}

        # 性能统计
        self.total_delay = 0
        self.total_energy = 0

    def initialize_trajectory(self, T):
        """
        初始化UAV的轨迹（位置和速度）
        :param T: 总时隙数
        """
        # 无人机初始轨迹+T个时隙的轨迹
        self.positions = np.zeros((T + 1, 3))
        self.velocities = np.zeros((T + 1, 3))

        # 无任务T个时隙的发射功率
        self.transmission_power = np.ones(T) * self.P_min

        # UAV初始轨迹
        # random.uniform，返回一个介于a和b之间的随机数，包括a和b
        self.positions[0] = np.array([
            random.uniform(0, self.area_size[0]),
            random.uniform(0, self.area_size[1]),
            random.uniform(0, self.area_size[2])
        ])

        self.velocities[0] = np.array([
            random.uniform(-self.v_max / 2, self.v_max / 2),
            random.uniform(-self.v_max / 2, self.v_max / 2),
            random.uniform(-self.v_max / 2, self.v_max / 2)
        ])

    def update_position(self, t):
        """
        更新UAV在时刻t的位置（方程1-3）
        :param t: 时间索引
        """
        if t >= len(self.positions) - 1:
            return

        # 位置更新（方程1）
        self.positions[t + 1] = self.positions[t] + self.velocities[t] * self.delta_t

        # 生成新速度，考虑加速度约束（方程3）
        acceleration = np.array([
            random.uniform(-self.a_max, self.a_max),
            random.uniform(-self.a_max, self.a_max),
            random.uniform(-self.a_max, self.a_max)
        ])

        new_v = self.velocities[t] + acceleration * self.delta_t

        # 速度约束（方程2）
        v_norm = np.linalg.norm(new_v)   # 计算无人机速度的模
        if v_norm > self.v_max:
            new_v = new_v * (self.v_max / v_norm)

        self.velocities[t + 1] = new_v

        # 位置边界约束，将数组中的元素限制在指定的范围内
        self.positions[t + 1] = np.clip(
            self.positions[t + 1],
            [0, 0, 0],
            [self.area_size[0], self.area_size[1], self.area_size[2]]
        )

    def calculate_transmission_rate(self, t, bs_pos, N_total):
        """
        计算UAV与BS之间的传输速率（方程4）
        Ru,bs(t) = (B/N) * log2(1 + Pu(t)|hu,bs(t)|^2 / σ^2)
        通信与感知融合，根据传感性能，更新传输速率
        Ru,bs(t) = (B/N) * log2(1 + Pu(t)|hu,bs(t)|^2 / (Pu(t)|eu(t)|^2 + σ^2))
        :param t: 时间索引
        :param bs_pos: 基站位置
        :param N_total: 总UAV数量
        :return: 传输速率 bps
        """
        h_estimated, channel_error = self.calculate_channel_gain(t, bs_pos)

        # 修正后的传输速率
        rate = (self.B / N_total) * np.log2(1 + (
                self.transmission_power[t] * np.abs(h_estimated) ** 2 /
                (self.transmission_power[t] * (channel_error ** 2) + self.sigma_squared)
        ))

        return rate

    def calculate_channel_gain(self, t, bs_pos):
        """
        计算信道增益和估计误差（方程5，11）
        基于感知精度计算真实信道增益和估计误差
        :param t: 时间索引
        :param bs_pos: 基站位置
        :return: (估计信道增益, 信道误差)
        """
        # 计算距离和路径损耗
        distance = self.get_distance_to_bs(t, bs_pos)

        # 自由空间路径损耗
        f_c = 2.4e9  # 载波频率 2.4GHz
        wavelength = 3e8 / f_c
        path_loss_db = 20 * np.log10(4 * np.pi * distance / wavelength)
        path_loss = 10 ** (-path_loss_db / 10)

        # 小尺度衰落（瑞利衰落）
        real_part = np.random.normal(0, 1 / np.sqrt(2))
        imag_part = np.random.normal(0, 1 / np.sqrt(2))
        small_scale = real_part + 1j * imag_part

        # 真实信道增益 h*_u(t)
        h_star = np.sqrt(path_loss) * small_scale

        # 基于位置感知误差计算信道估计误差
        # 感知误差影响信道估计精度
        velocity_norm = np.linalg.norm(self.velocities[t])
        sensing_error_factor = velocity_norm * 0.01  # 简化的感知误差模型

        # 信道估计误差 (方程11)
        channel_error_real = np.random.normal(0, sensing_error_factor)
        channel_error_imag = np.random.normal(0, sensing_error_factor)
        channel_error = channel_error_real + 1j * channel_error_imag

        # 估计信道增益 hu(t) = h*u(t) + error
        h_estimated = h_star + channel_error

        # 返回估计增益和误差 (用于方程11)
        error_magnitude = np.abs(channel_error)

        return h_estimated, error_magnitude

    def calculate_round_trip_delay(self, t, bs_pos):
        """
        计算往返延迟 τu (方程8)
        :param t: 时间索引
        :param bs_pos: 基站位置
        :return: 往返延迟
        """
        distance = self.get_distance_to_bs(t, bs_pos)
        return 2 * distance / 3e8  # c = 3e8 m/s

    def get_distance_to_bs(self, t, bs_pos):
        """
        计算到基站的距离（用于感知模型方程8）
        :param t: 时间索引
        :param bs_pos: 基站位置
        :return: 距离
        """
        return np.linalg.norm(self.positions[t] - bs_pos)   # 计算向量的模

    def generate_computation_task(self, t):
        """
        生成UAV的计算任务（通信）
        :param t: 时间索引
        :return: 任务字典或None
        """
        if random.random() < 0.5:  # 50%概率生成计算任务
            task = {
                'task_id': f"comp_{self.uav_id}_{t}",
                'type': 'computation',
                'uav_id': self.uav_id,
                'time_slot': t,
                'data_size': random.uniform(0.1, 1.0) * 1e6,  # 数据大小 (决定了任务卸载时延，方程17)
                'cpu_cycles': random.uniform(5, 50) * 1e8,  # CPU周期 (决定了本地或MEC的任务计算时延，方程14,19)
                'deadline': t + random.randint(3, 8),  # 截止时间
                'offload_decision': 1,  # 1=本地处理, 0=卸载，默认本地处理
            }
            self.computation_tasks[t] = task
            return task
        return None

    def calculate_local_processing_time(self, task):
        """
        计算本地处理时间（方程14）
        tlocal,u(t) = Ccomp,u(t) / flocal,u
        :param task: 任务信息
        :return: 处理时间
        """
        if task['type'] == 'computation':
            return task['cpu_cycles'] / self.f_local
        else:
            return float('inf')  # 其他任务类型不能本地处理

    def calculate_local_energy(self, task):
        """
        计算本地处理能耗（方程15-16）
        Elocal,u(t) = λ(flocal,u)^2 * Ccomp,u(t)
        :param task: 任务信息
        :return: 能耗 J
        """
        if task['type'] == 'computation':
            # 每个CPU周期所需的能耗 * 计算任务CPU周期
            return self.lambda_local * (self.f_local ** 2) * task['cpu_cycles']
        else:
            return float('inf')

    def calculate_uplink_transmission_time(self, task, t, bs_pos, N_total):
        """
        计算上行传输时间（方程17）
        ttrans_up,u(t) = Dcomp,u(t) / Ru(t)
        :param task: 任务信息
        :param t: 时间索引
        :param bs_pos: 基站位置
        :param N_total: 总UAV数量
        :return: 传输时间 s
        """
        if task['type'] == 'computation' and task['offload_decision'] == 0:
            rate = self.calculate_transmission_rate(t, bs_pos, N_total)
            return task['data_size'] / rate
        return 0

    def calculate_uplink_energy(self, task, t, bs_pos, N_total):
        """
        计算上行传输能耗（方程18）
        Eup,u(t) = Pu * ttrans_up,u(t)
        :param task: 任务信息
        :param t: 时间索引
        :param bs_pos: 基站位置
        :param N_total: 总UAV数量
        :return: 传输能耗 J
        """
        transmission_time = self.calculate_uplink_transmission_time(task, t, bs_pos, N_total)
        return self.transmission_power[t] * transmission_time

    def set_power(self, t, power):
        """
        设置无人机发射功率，该值随时间变化
        :param t: 时间索引
        :param power: 功率值 W
        """
        # 将超出给定范围的值裁剪到范围的边界值
        if t < len(self.transmission_power):
            self.transmission_power[t] = np.clip(power, self.P_min, self.P_max)


class BaseStation:
    """
    基站类 - 负责主动感知和通信中继
    """

    def __init__(self, position, is_debug=False):
        """
        初始化基站
        :param position: 基站位置
        :param is_debug: 调试模式
        """
        self.position = np.array(position)
        self.is_debug = is_debug

        # 感知参数
        self.f0 = 1  # 基本感知频率 Hz
        self.k_sen = 0.05  # 调节速度对传感频率影响程度的系数
        self.f_max = 10  # 最大感知频率 Hz

    def generate_sensing_tasks(self, uavs, t):
        """
        基站主动生成感知任务（针对所有UAV）
        这些任务是基站为了感知UAV位置而产生的
        :param uavs: UAV列表
        :param t: 时间索引
        :return: 感知任务列表
        """
        sensing_tasks = []

        for uav in uavs:
            # 计算对该UAV的感知频率
            f_sen = self.calculate_sensing_frequency(uav.velocities[t])
            sen_count = int(f_sen)

            # 为该UAV生成感知任务
            for sen_idx in range(sen_count):
                task = {
                    'task_id': f"sensing_{uav.uav_id}_{t}_{sen_idx}",
                    'type': 'sensing',
                    'target_uav_id': uav.uav_id,  # 被感知的UAV
                    'time_slot': t,
                    'sen_index': sen_idx,
                    # 感知任务的计算量（处理回波信号）
                    'data_size': 0.02 * 1e6,  # 感知数据量较小
                    'cpu_cycles': 1e8,  # 信号处理计算量
                    'deadline': t + 1,  # 感知任务需要快速处理
                    'round_trip_delay': uav.calculate_round_trip_delay(t, self.position),  # 感知信号的往返时延
                }
                sensing_tasks.append(task)

        return sensing_tasks

    def calculate_sensing_frequency(self, uav_velocity):
        """
        计算对UAV的感知频率（方程12-13）
        fsen,u(t) = f0 + k * ||vu(t)||, fsen,u(t) ≤ fmax
        :param uav_velocity: UAV速度向量
        :return: 感知频率
        """
        # 计算向量长度
        velocity_norm = np.linalg.norm(uav_velocity)
        f_sen = self.f0 + self.k_sen * velocity_norm
        return min(f_sen, self.f_max)


class MEC:
    """
    移动边缘计算服务器类 - 负责处理感知任务和卸载的计算任务
    """

    def __init__(self, capacity=1e11, max_queue_length=100, is_debug=False):
        """
        初始化MEC服务器
        :param capacity: 处理能力 Hz
        :param max_queue_length: 最大队列长度
        :param is_debug: 调试模式
        """
        self.capacity = capacity
        self.max_queue_length = max_queue_length
        self.is_debug = is_debug

        # 队列管理
        self.queue_length = None  # 数据队列长度（字节）
        self.queued_tasks = []  # 任务队列

        # 优先级权重（方程28）
        self.omega1 = 1 / 3  # 截止时间权重
        self.omega2 = 1 / 3  # 速度权重
        self.omega3 = 1 / 3  # 等待时间权重

        # 处理优先级靠前的前k个任务
        self.k = 5

        # 统计信息
        self.processed_tasks = 0
        self.total_processing_time = 0

    def calculate_total_processing_time(self, task):
        """
        计算任务的总处理时间（方程19，24）
        - 计算任务：tmec,u(t) = tqueue,u(t) + Ccomp,u(t)/fmec,u(t)
        - 感知任务：tsen_total,u(t) = τu + tqueue,u(t) + Csen,u(t)/fmec,u(t)
        :param task: 任务信息
        :return: 总处理时间
        """
        total_processing_time = 0

        if task['type'] == 'sensing':
            # 感知任务加上往返延迟
            total_processing_time += task.get('round_trip_delay', 0)
            
        # 获取队列等待时间
        total_processing_time += task.get('queue_wait_time', 0)

        # 获取MEC处理时间
        if 'processing_time' in task:
            total_processing_time += task['processing_time']

        # 返回总处理时间
        return total_processing_time

    def initialize_queue(self, T):
        """
        初始化任务队列，每个时隙的任务队列长度是不同的
        :param T: 总时隙数
        """
        self.queue_length = np.zeros(T + 1)

    def update_queue_length(self, t, arrived_tasks, completed_tasks):
        """
        更新队列长度（方程27）
        Q(t+1) = [Q(t) + Σ[(1-αu(t))(1-ru(t))Dcomp,u(t) + αu(t)Dsen,u(t)] - C(t)]+
        :param t: 当前时隙
        :param arrived_tasks: 到达任务
        :param completed_tasks: 完成任务
        """
        # 计算新到达的数据量
        arrived_data = 0
        for task in arrived_tasks:
            if task['type'] == 'sensing':  # αu(t)=1的感知任务
                arrived_data += task.get('data_size', 0)
            elif task['type'] == 'computation' and task['offload_decision'] == 0:  # αu(t)=0, ru(t)=0的计算任务
                arrived_data += task.get('data_size', 0)

        # 计算完成的数据量
        completed_data = sum(task.get('data_size', 0) for task in completed_tasks)

        # 更新队列长度（方程27）
        if t + 1 < len(self.queue_length):
            self.queue_length[t + 1] = max(0, min(
                self.queue_length[t] + arrived_data - completed_data,
                self.max_queue_length
            ))

    def calculate_task_priority(self, task, wait_time, uav_velocity):
        """
        计算任务优先级（方程28）
        Fu(t) = ω1(t)·1/(Tmax_u(t)-t) + ω2(t)·||vu(t)|| + ω3(t)·twait,u(t)
        :param task: 任务信息
        :param wait_time: 等待时间
        :param uav_velocity: UAV速度
        :return: 优先级值
        """
        t = task['time_slot']
        deadline = task['deadline'] - t

        # 截止时间因子
        if deadline <= 0:
            deadline_factor = float('inf')
        else:
            deadline_factor = 1 / deadline

        # 速度因子
        velocity_norm = np.linalg.norm(uav_velocity)

        # 优先级计算
        priority = (
                self.omega1 * deadline_factor +
                self.omega2 * velocity_norm +
                self.omega3 * wait_time
        )

        return priority

    def schedule_tasks(self, arrived_tasks, t, uav_states):
        """
        任务调度（基于优先级），这个过程产生MEC处理时间和队列等待时间
        :param arrived_tasks: 当前到达的任务（包括感知任务和卸载的计算任务）
        :param t: 当前时隙
        :param uav_states: UAV状态信息（位置、速度等）
        :return: (已完成任务, 队列中任务)
        """
        # 添加新任务到队列
        for task in arrived_tasks:
            # 感知任务总是需要MEC处理
            # 计算任务只有卸载时才需要MEC处理
            if task['type'] == 'sensing' or (task['type'] == 'computation' and task['offload_decision'] == 0):
                task['arrival_time'] = t  # 记录任务到达时间
                task['wait_time'] = 0
                self.queued_tasks.append(task)

        # 更新所有队列中任务的等待时间
        for task in self.queued_tasks:
            task['wait_time'] = t - task.get('arrival_time', t)

        # 计算优先级并排序
        task_priorities = []
        for task in self.queued_tasks:
            if task['type'] == 'sensing':
                target_uav_id = task['target_uav_id']
                uav_state = uav_states.get(target_uav_id, {'velocity': np.zeros(3)})
            else:
                uav_id = task['uav_id']
                uav_state = uav_states.get(uav_id, {'velocity': np.zeros(3)})

            priority = self.calculate_task_priority(
                task,
                task['wait_time'],
                uav_state['velocity']
            )
            task_priorities.append((task, priority))

        # 按优先级排序（优先级高的先处理）
        sorted_tasks = [task for task, _ in sorted(task_priorities, key=lambda x: x[1], reverse=True)]

        # 处理任务（分配处理能力）
        completed_tasks = []
        remaining_tasks = []
        used_capacity = 0

        # 只处理前 k 个任务
        priority_tasks = sorted_tasks[:self.k] if len(sorted_tasks) > self.k else sorted_tasks

        for task in sorted_tasks:
            if task in priority_tasks and used_capacity < self.capacity:
                # 为优先任务分配足够的处理能力
                required_freq = task['cpu_cycles'] / 1.0  # 使任务在1个时隙内完成所需的频率

                if required_freq <= (self.capacity - used_capacity):
                    # 如果有足够的剩余容量
                    allocated_freq = required_freq

                    # 记录开始处理时间和队列等待时间
                    task['queue_wait_time'] = task['wait_time']  # 记录从到达到开始处理的等待时间

                    # 计算MEC处理时间
                    processing_time = task['cpu_cycles'] / allocated_freq

                    # 检查是否能在当前时隙完成
                    if processing_time <= 1.0:  # 一个时隙内能完成
                        task['completion_time'] = t
                        task['processing_time'] = processing_time
                        task['allocated_frequency'] = allocated_freq
                        completed_tasks.append(task)
                        used_capacity += allocated_freq
                        self.processed_tasks += 1
                    else:
                        remaining_tasks.append(task)
                else:
                    # 剩余容量不足
                    remaining_tasks.append(task)
            else:
                # 非优先任务或容量用尽
                remaining_tasks.append(task)

        # 更新队列
        self.queued_tasks = remaining_tasks

        if self.is_debug and (completed_tasks or arrived_tasks):
            print(
                f"  MEC时隙{t}: 到达{len(arrived_tasks)}个任务, 完成{len(completed_tasks)}个任务, 队列剩余{len(remaining_tasks)}个任务")

        return completed_tasks, remaining_tasks


class UAVMECSystem:
    """
    UAV-MEC系统协调类 - 负责系统级的协调和优化
    """

    def __init__(self, N, T, area_size=(1000, 1000, 100), is_debug=True):
        """
        初始化系统
        :param N: UAV数量
        :param T: 时隙数
        :param area_size: 区域大小
        :param is_debug: 调试模式
        """
        self.N = N
        self.T = T
        self.area_size = area_size
        self.is_debug = is_debug

        # 本地处理权重 (延迟权重, 能耗权重)
        self.alpha_local_delay = 0.5  # 本地处理延迟权重
        self.alpha_local_energy = 0.5  # 本地处理能耗权重

        # 卸载处理权重 (延迟权重, 能耗权重)
        self.alpha_offload_delay = 0.5  # 卸载处理延迟权重
        self.alpha_offload_energy = 0.5  # 卸载处理能耗权重

        # 创建基站
        bs_position = [area_size[0] / 2, area_size[1] / 2, 4]
        self.base_station = BaseStation(bs_position, is_debug)

        # 创建UAV实例
        self.uavs = []
        for i in range(N):
            uav = UAV(i, area_size, is_debug)
            uav.initialize_trajectory(T)
            self.uavs.append(uav)

        # 创建MEC实例
        self.mec = MEC(is_debug=is_debug)
        self.mec.initialize_queue(T)

        # 预存储所有时隙的感知任务
        self.sensing_tasks_by_timeslot = {}

        # 系统性能统计
        self.total_delay = 0
        self.total_energy = 0
        self.convergence_history = {
            'iterations': [],
            'total_delays': [],
            'total_energies': [],
            'best_delays': [],
            'best_energies': []
        }

    def update_all_uav_positions(self):
        """
        更新所有UAV位置轨迹
        """
        for t in range(self.T):
            for uav in self.uavs:
                uav.update_position(t)

    def generate_all_tasks(self):
        """
        生成所有任务
        - UAV生成计算任务
        - 基站生成感知任务
        """
        for t in range(self.T):
            # UAV生成计算任务
            for uav in self.uavs:
                uav.generate_computation_task(t)

            # 基站生成感知任务
            sensing_tasks = self.base_station.generate_sensing_tasks(self.uavs, t)
            self.sensing_tasks_by_timeslot[t] = sensing_tasks

    def get_tasks_for_timeslot(self, t):
        """
        获取时隙t的所有任务
        :param t: 时隙
        :return: (计算任务列表, 感知任务列表)
        """
        # UAV的计算任务
        computation_tasks = []
        for uav in self.uavs:
            if t in uav.computation_tasks:
                computation_tasks.append(uav.computation_tasks[t])

        # 基站的感知任务
        sensing_tasks = self.sensing_tasks_by_timeslot.get(t, [])

        return computation_tasks, sensing_tasks

    def calculate_timeslot_cost(self, computation_tasks, t):
        """
        计算单个时隙的计算任务成本（方程21，23）
        使用不同的权重参数计算本地处理和卸载处理的成本
        :param computation_tasks: 计算任务列表
        :param t: 时隙
        :return: (总成本, 总时延, 总能耗)
        """
        total_cost = 0
        total_delay = 0
        total_energy = 0

        for task in computation_tasks:
            uav = self.uavs[task['uav_id']]

            if task['offload_decision'] == 1:  # 本地处理（方程21）
                delay = uav.calculate_local_processing_time(task)
                energy = uav.calculate_local_energy(task)
                # 使用本地处理权重
                task_cost = self.alpha_local_delay * delay + self.alpha_local_energy * energy

            else:  # 卸载到MEC（方程23）
                # 上行传输时间和能耗
                delay = uav.calculate_uplink_transmission_time(
                    task, t, self.base_station.position, self.N
                )

                # 估算MEC处理时延
                estimated_mec_delay = self.estimate_mec_processing_delay(task, t)
                delay += estimated_mec_delay

                energy = uav.calculate_uplink_energy(
                    task, t, self.base_station.position, self.N
                )

                # 使用卸载处理权重
                task_cost = self.alpha_offload_delay * delay + self.alpha_offload_energy * energy

            total_cost += task_cost
            total_delay += delay
            total_energy += energy

        return total_cost, total_delay, total_energy

    def estimate_mec_processing_delay(self, task, t):
        """
        估算MEC处理延迟（用于成本评估阶段）
        有多种估算策略可选择
        :param task: 任务信息
        :param t: 当前时隙
        :return: 估算的处理延迟
        """
        # 策略1: 基于当前队列状态的估算
        return self._estimate_based_on_queue_state(task, t)

        # 策略2: 基于历史平均的估算
        # return self._estimate_based_on_history(task, t)

    def _estimate_based_on_queue_state(self, task, t):
        """策略1: 基于当前队列状态的估算"""
        # 1. 估算队列等待时间
        if t < len(self.mec.queue_length):
            current_queue_length = self.mec.queue_length[t]
            # 根据当前队列长度和处理能力估算等待时间
            estimated_queue_wait = current_queue_length / (self.mec.capacity * 0.8)  # 考虑80%利用率
        else:
            estimated_queue_wait = 0

        # 2. 估算纯处理时间
        estimated_processing_time = task['cpu_cycles'] / self.mec.capacity

        # 3. 考虑调度和优先级影响
        # 如果当前有很多任务在队列中，新任务可能需要等待
        queue_congestion_factor = 1.0
        if len(self.mec.queued_tasks) > self.mec.k:
            queue_congestion_factor = 1.2 + 0.1 * (len(self.mec.queued_tasks) - self.mec.k)

        total_delay = (estimated_queue_wait + estimated_processing_time) * queue_congestion_factor
        return total_delay

    def _estimate_based_on_history(self, task, t):
        """策略2: 基于历史平均处理时间的估算"""
        if self.mec.processed_tasks > 0:
            avg_processing_time = self.mec.total_processing_time / self.mec.processed_tasks
            return avg_processing_time
        else:
            # 如果没有历史数据，回退到基于队列状态的估算
            return self._estimate_based_on_queue_state(task, t)

    def optimize_joint_variables(self, computation_tasks, t):
        """
        联合优化卸载决策和发射功率（仅针对计算任务）
        :param computation_tasks: 计算任务列表
        :param t: 时隙
        """
        if not computation_tasks:
            return

        if self.is_debug:
            print(f"  优化时隙{t}的{len(computation_tasks)}个计算任务")

        # 暴力搜索最优组合
        best_cost = float('inf')
        best_decisions = None
        best_powers = None

        # 所有卸载决策组合
        offload_combinations = list(product([0, 1], repeat=len(computation_tasks)))

        # 所有功率组合
        power_combinations = list(product(self.uavs[0].power_options, repeat=self.N))

        for offload_combo in offload_combinations:
            for power_combo in power_combinations:
                # 设置决策和功率
                for i, task in enumerate(computation_tasks):
                    task['offload_decision'] = offload_combo[i]

                for i, uav in enumerate(self.uavs):
                    uav.set_power(t, power_combo[i])

                # 计算成本（使用新的成本函数）
                cost, _, _ = self.calculate_timeslot_cost(computation_tasks, t)

                if cost < best_cost:
                    best_cost = cost
                    best_decisions = offload_combo
                    best_powers = power_combo

        # 应用最佳决策
        if best_decisions and best_powers:
            for i, task in enumerate(computation_tasks):
                task['offload_decision'] = best_decisions[i]

            for i, uav in enumerate(self.uavs):
                uav.set_power(t, best_powers[i])

            if self.is_debug:
                print(f"    最佳卸载决策: {best_decisions}")
                print(f"    最佳功率分配: {[f'{p:.3f}W' for p in best_powers]}")

    def process_timeslot(self, t):
        """
        计算当前时隙中，通信+感知的时延，以及通信的能耗
        :param t: 时隙
        :return: (总成本, 总时延, 总能耗)
        """
        # 获取当前任务
        computation_tasks, sensing_tasks = self.get_tasks_for_timeslot(t)
        all_tasks = computation_tasks + sensing_tasks

        if self.is_debug:
            print(f"时隙{t}: {len(computation_tasks)}个计算任务, {len(sensing_tasks)}个感知任务")

        # 联合优化计算任务的卸载决策和功率分配
        self.optimize_joint_variables(computation_tasks, t)

        # 准备UAV状态信息用于MEC调度
        uav_states = {}
        for uav in self.uavs:
            uav_states[uav.uav_id] = {
                'position': uav.positions[t],
                'velocity': uav.velocities[t]
            }

        # 感知任务中的target_uav_id也需要状态信息
        for task in sensing_tasks:
            target_uav_id = task['target_uav_id']
            if target_uav_id not in uav_states:
                target_uav = self.uavs[target_uav_id]
                uav_states[target_uav_id] = {
                    'position': target_uav.positions[t],
                    'velocity': target_uav.velocities[t]
                }

        # MEC任务调度（处理感知任务和卸载的计算任务）
        completed_tasks, remaining_tasks = self.mec.schedule_tasks(all_tasks, t, uav_states)

        # 更新MEC队列
        self.mec.update_queue_length(t, all_tasks, completed_tasks)

        # 计算计算任务的成本、时延和能耗
        comp_cost, comp_delay, comp_energy = self.calculate_timeslot_cost(computation_tasks, t)

        # 感知任务的时延（主要是MEC处理延迟，能耗由基站承担，UAV无额外能耗）
        sensing_delay = 0

        for task in sensing_tasks:
            if task in completed_tasks:
                # 使用完整记录的总处理时间
                sensing_delay += self.mec.calculate_total_processing_time(task)
            else:
                # 估算感知任务延迟
                sensing_delay += task.get('round_trip_delay', 0) + 0.001  # 1ms处理时间

        sensing_cost = 0.5 * sensing_delay  # 感知任务成本

        # 汇总结果
        total_cost = comp_cost + sensing_cost
        total_delay = comp_delay + sensing_delay
        total_energy = comp_energy

        if self.is_debug:
            print(f"  计算任务: 成本={comp_cost:.4f}, 时延={comp_delay:.4f}s, 能耗={comp_energy:.4f}J")
            print(f"  感知任务: 成本={sensing_cost:.4f}, 时延={sensing_delay:.4f}s")
            print(f"  总计: 成本={total_cost:.4f}, 时延={total_delay:.4f}s, 能耗={total_energy:.4f}J")

        return total_cost, total_delay, total_energy

    def run_optimization(self):
        """
        运行系统优化（方程29）
        目标：min Σ Σ (ttotal,u(t) + Etotal,u(t))
        :return: (总成本, 总延迟, 总能耗)
        """
        print("开始UAV-MEC系统联合优化...")

        # 系统初始化
        self.update_all_uav_positions()
        self.generate_all_tasks()

        total_cost = 0

        # 逐时隙处理
        for t in range(self.T):
            cost, delay, energy = self.process_timeslot(t)
            total_cost += cost
            self.total_delay += delay
            self.total_energy += energy

            # 记录收敛历史
            self.convergence_history['iterations'].append(t)
            self.convergence_history['total_delays'].append(self.total_delay)
            self.convergence_history['total_energies'].append(self.total_energy)
            self.convergence_history['best_delays'].append(self.total_delay)
            self.convergence_history['best_energies'].append(self.total_energy)

        print(f"优化完成! 总成本: {total_cost:.4f}, 总延迟: {self.total_delay:.4f}s, 总能耗: {self.total_energy:.4f}J")
        return total_cost, self.total_delay, self.total_energy

    def get_statistics(self):
        """
        获取系统统计信息
        :return: 统计字典
        """
        total_comp_tasks = sum(len(uav.computation_tasks) for uav in self.uavs)
        total_sen_tasks = self.mec.processed_tasks - sum(
            1 for uav in self.uavs for t in uav.computation_tasks
            if uav.computation_tasks[t]['offload_decision'] == 0
        )

        return {
            'total_uavs': self.N,
            'total_timeslots': self.T,
            'total_computation_tasks': total_comp_tasks,
            'total_sensing_tasks': total_sen_tasks,
            'total_delay': self.total_delay,
            'total_energy': self.total_energy,
            'mec_processed_tasks': self.mec.processed_tasks,
            'final_queue_length': self.mec.queue_length[-1]
        }


def main(N=2, T=3):
    """
    主函数
    :param N: UAV数量
    :param T: 时隙数
    """
    # 创建系统
    system = UAVMECSystem(N, T, is_debug=True)

    # 运行优化
    total_cost, total_delay, total_energy = system.run_optimization()

    # 打印统计信息
    stats = system.get_statistics()
    print("\n=== 系统统计信息 ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    return total_cost, total_delay, total_energy


if __name__ == '__main__':
    main(N=5, T=5)