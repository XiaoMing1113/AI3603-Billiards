import math
import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*invalid value encountered in divide.*",
    module=r"pooltool\.ptmath\.roots\.core",
)
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from .agent import Agent


# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass

def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")

def simulate_with_timeout(shot, timeout=3):
    """带超时保护的物理模拟

    参数：
        shot: pt.System 对象
        timeout: 超时时间（秒），默认3秒

    返回：
        bool: True 表示模拟成功，False 表示超时或失败

    说明：
        使用 signal.SIGALRM 实现超时机制（仅支持 Unix/Linux）
        Windows 下暂不支持超时中断，将直接运行模拟
    """
    # Windows 不支持 SIGALRM
    if not hasattr(signal, "SIGALRM"):
        try:
            pt.simulate(shot, inplace=True)
            return True
        except Exception as e:
            raise e

    # 设置超时信号处理器
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)  # 设置超时时间

    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)  # 取消超时
        return True
    except SimulationTimeoutError:
        print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
        return False
    except Exception as e:
        signal.alarm(0)  # 取消超时
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器

# ============================================


def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）

    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...] 或 ['8']

    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -900（非法黑8）, -1200（白球+黑8）, -30（首球/碰库犯规）

    规则核心：
        - 清台前：player_targets = ['1'-'7'] 或 ['9'-'15']，黑8不属于任何人
        - 清台后：player_targets = ['8']，黑8成为唯一目标球
    """
    # 0. 提取事件信息
    events = shot.events

    cue_pocketed = False
    eight_pocketed = False
    own_pocketed = []
    enemy_pocketed = []

    # 检查进袋球
    for ball_id, ball in shot.balls.items():
        if ball.state.s == 4:  # s=4 表示进袋
            # 必须是新进的球（上一状态不在袋中）
            if ball_id in last_state and last_state[ball_id].state.s != 4:
                if ball_id == "cue":
                    cue_pocketed = True
                elif ball_id == "8":
                    eight_pocketed = True
                elif ball_id in player_targets:
                    own_pocketed.append(ball_id)
                else:
                    enemy_pocketed.append(ball_id)

    # 1. 犯规检测 (简化版)
    # 首球犯规：白球碰到的第一个球不是自己的目标球（清台前黑8算对方）
    foul_first_hit = False
    foul_no_rail = False  # 暂未详细实现碰库检测，仅作为惩罚项保留接口

    # 接触检测：白球是否碰到球
    first_contact_ball_id = None
    cue_hit_cushion = False  # 白球是否先撞库
    target_hit_cushion = False  # 目标球是否撞库（进袋前）

    # 解析 pooltool 事件
    # 注意：pooltool 的 events 列表按时间排序
    # 我们只关心白球的第一次碰撞
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, "ids") else []

        # 检查是否是白球碰撞
        if "collision" in et and "cue" in ids:
            other_ids = [i for i in ids if i != "cue"]
            if not other_ids:
                continue  # 可能是白球碰库

            # 这是一个球-球碰撞
            other_id = other_ids[0]
            # 库边碰撞通常没有 ball id 或者 id 是 cushion
            # pooltool 中 cushion collision event type 是 'ball-cushion collision'

            first_contact_ball_id = other_id
            break  # 找到首碰，退出

    # 判定首球是否合法
    if first_contact_ball_id is None:
        # 没碰到任何球
        foul_first_hit = True
    else:
        # 碰到了球，检查是否是目标球
        if first_contact_ball_id not in player_targets:
            # 特殊情况：如果是开球或未分球状态，player_targets 可能包含除了8以外的所有球
            # 这里假设 player_targets 已经正确设置
            foul_first_hit = True
    illegal_eight_first_hit = first_contact_ball_id == "8" and player_targets != ["8"]

    # 检查是否有进球（任何球）
    new_pocketed = [
        bid
        for bid, b in shot.balls.items()
        if b.state.s == 4 and (bid not in last_state or last_state[bid].state.s != 4)
    ]

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True

    # 4. 计算奖励分数
    score = 0

    # 白球进袋处理
    if cue_pocketed and eight_pocketed:
        score -= 1200  # 白球+黑8同时进袋，严重犯规
    elif cue_pocketed:
        score -= 100  # 白球进袋
    elif eight_pocketed:
        # 黑8进袋：只有清台后（player_targets == ['8']）才合法
        if player_targets == ['8']:
            score += 100  # 合法打进黑8
        else:
            score -= 900  # 清台前误打黑8，判负

    if illegal_eight_first_hit:
        score -= 200

    # 首球犯规和碰库犯规
    if foul_first_hit:
        score -= 50
    if foul_no_rail:
        score -= 30

    # 进球得分（own_pocketed 已根据 player_targets 正确分类）
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20

    # 合法无进球小奖励
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10

    # 2. 接触奖励：白球击中自己的目标球
    first_contact_ball_id = None
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, "ids") else []
        if ("cushion" not in et) and ("pocket" not in et) and ("cue" in ids):
            other_ids = [i for i in ids if i != "cue"]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break

    if first_contact_ball_id in player_targets:
        score += 20.0  # 击中目标球给予额外奖励 (原为 5.0)

    # 3. 距离奖励（简化）：如果击球后目标球停在桌面上，根据其位置给予微小奖励
    # 这里略过，因为需要复杂的口袋位置计算，且容易导致Agent过度关注位置而忽略进球

    return score


def analyze_shot_for_reward_agent(
    shot: pt.System, last_state: dict, player_targets: list
):
    events = shot.events
    cue_pocketed = False
    eight_pocketed = False
    own_pocketed = []
    enemy_pocketed = []
    for ball_id, ball in shot.balls.items():
        if ball.state.s == 4:
            if ball_id in last_state and last_state[ball_id].state.s != 4:
                if ball_id == "cue":
                    cue_pocketed = True
                elif ball_id == "8":
                    eight_pocketed = True
                elif ball_id in player_targets:
                    own_pocketed.append(ball_id)
                else:
                    enemy_pocketed.append(ball_id)
    first_contact_ball_id = None
    valid_ball_ids = {
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
    }
    for e in events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, "ids") else []
        if ("cushion" not in et) and ("pocket" not in et) and ("cue" in ids):
            other_ids = [i for i in ids if i != "cue" and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    cue_hit_cushion = False
    target_hit_cushion = False
    for e in events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, "ids") else []
        if "cushion" in et:
            if "cue" in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True
    new_pocketed = [
        bid
        for bid, b in shot.balls.items()
        if b.state.s == 4 and (bid in last_state and last_state[bid].state.s != 4)
    ]
    remaining_own_before = [
        bid for bid in player_targets if last_state[bid].state.s != 4
    ]
    foul_first_hit = False
    foul_no_rail = False
    if first_contact_ball_id is None:
        foul_first_hit = True
    else:
        if len(remaining_own_before) == 0:
            if first_contact_ball_id != "8":
                foul_first_hit = True
        else:
            if first_contact_ball_id not in player_targets:
                foul_first_hit = True
    if (
        len(new_pocketed) == 0
        and first_contact_ball_id is not None
        and (not cue_hit_cushion)
        and (not target_hit_cushion)
    ):
        foul_no_rail = True
    score = 0
    if cue_pocketed and eight_pocketed:
        score -= 1200
    elif cue_pocketed:
        score -= 70
    elif eight_pocketed:
        if len(remaining_own_before) == 0:
            score += 300
        else:
            score -= 900
    if first_contact_ball_id == "8" and len(remaining_own_before) != 0:
        score -= 200
    if foul_first_hit:
        score -= 50
    if foul_no_rail:
        score -= 30
    score += len(own_pocketed) * 100
    score -= len(enemy_pocketed) * 20
    if first_contact_ball_id in player_targets:
        score += 20.0
    if (
        score == 0
        and not cue_pocketed
        and not eight_pocketed
        and not foul_first_hit
        and not foul_no_rail
    ):
        score = 10
    return score


class Agent():
    """Agent 基类"""
    def __init__(self):
        pass

    def decision(self, *args, **kwargs):
        """决策方法（子类需实现）
        
        返回：dict, 包含 'V0', 'phi', 'theta', 'a', 'b'
        """
        pass

    def _random_action(self,):
        """生成随机击球动作
        
        返回：dict
            V0: [0.5, 8.0] m/s
            phi: [0, 360] 度
            theta: [0, 90] 度
            a, b: [-0.5, 0.5] 球半径比例
        """
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),   # 初速度 0.5~8.0 m/s
            'phi': round(random.uniform(0, 360), 2),    # 水平角度 (0°~360°)
            'theta': round(random.uniform(0, 90), 2),   # 垂直角度
            'a': round(random.uniform(-0.5, 0.5), 3),   # 杆头横向偏移（单位：球半径比例）
            'b': round(random.uniform(-0.5, 0.5), 3)    # 杆头纵向偏移
        }
        return action


class BasicAgent(Agent):
    """基于贝叶斯优化的智能 Agent"""

    def __init__(self, target_balls=None):
        """初始化 Agent
        
        参数：
            target_balls: 保留参数，暂未使用
        """
        super().__init__()

        # 搜索空间
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90), 
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }

        # 优化参数
        self.INITIAL_SEARCH = 20
        self.OPT_SEARCH = 10
        self.ALPHA = 1e-2

        # 模拟噪声（可调整以改变训练难度）
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        self.enable_noise = False
        
        print("BasicAgent (贝叶斯优化版) 已初始化。")

    def _create_optimizer(self, reward_function, seed):
        """创建贝叶斯优化器
        
        参数：
            reward_function: 目标函数，(V0, phi, theta, a, b) -> score
            seed: 随机种子
        
        返回：
            BayesianOptimization对象
        """
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=10,
            random_state=seed
        )

        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
            gamma_pan=1.0
        )

        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer
        )
        optimizer._gp = gpr

        return optimizer

    def decision(self, balls=None, my_targets=None, table=None):
        """使用贝叶斯优化搜索最佳击球参数
        
        参数：
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象
        
        返回：
            dict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
                失败时返回随机动作
        """
        if balls is None:
            print(f"[BasicAgent] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()
        try:

            # 保存一个击球前的状态快照，用于对比
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[BasicAgent] 我的目标球已全部清空，自动切换目标为：8号球")

            # 1.动态创建"奖励函数" (Wrapper)
            # 贝叶斯优化器会调用此函数，并传入参数
            def reward_fn_wrapper(V0, phi, theta, a, b):
                # 创建一个用于模拟的沙盒系统
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")

                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

                try:
                    if self.enable_noise:
                        V0_noisy = V0 + np.random.normal(0, self.noise_std['V0'])
                        phi_noisy = phi + np.random.normal(0, self.noise_std['phi'])
                        theta_noisy = theta + np.random.normal(0, self.noise_std['theta'])
                        a_noisy = a + np.random.normal(0, self.noise_std['a'])
                        b_noisy = b + np.random.normal(0, self.noise_std['b'])

                        V0_noisy = np.clip(V0_noisy, 0.5, 8.0)
                        phi_noisy = phi_noisy % 360
                        theta_noisy = np.clip(theta_noisy, 0, 90)
                        a_noisy = np.clip(a_noisy, -0.5, 0.5)
                        b_noisy = np.clip(b_noisy, -0.5, 0.5)

                        shot.cue.set_state(V0=V0_noisy, phi=phi_noisy, theta=theta_noisy, a=a_noisy, b=b_noisy)
                    else:
                        shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)

                    # 关键：使用带超时保护的物理模拟（3秒上限）
                    if not simulate_with_timeout(shot, timeout=3):
                        return 0  # 超时是物理引擎问题，不惩罚agent
                except Exception as e:
                    # 模拟失败，给予极大惩罚
                    return -500
                
                # 使用我们的"裁判"来打分
                score = analyze_shot_for_reward(
                    shot=shot, last_state=last_state_snapshot, player_targets=my_targets
                )
                return score

            # 2.运行贝叶斯优化
            # 随机种子设为当前时间戳微秒数，保证每次不一样
            seed = int(datetime.now().microsecond)
            optimizer = self._create_optimizer(reward_fn_wrapper, seed)

            optimizer.maximize(
                init_points=self.INITIAL_SEARCH,
                n_iter=self.OPT_SEARCH,
            )

            # 3.获取最佳参数
            best_params = optimizer.max["params"]
            best_score = optimizer.max["target"]

            print(
                f"[BasicAgent] 思考完毕。最佳分数: {best_score:.1f}, 参数: V0={best_params['V0']:.2f}, phi={best_params['phi']:.2f}"
            )

            return best_params

        except Exception as e:
            print(f"[BasicAgent] 决策出错: {e}，回退到随机动作。")
            import traceback
            traceback.print_exc()
            return self._random_action()
