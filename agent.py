"""
agent.py - Agent 决策模块

定义 Agent 基类和具体实现：
- Agent: 基类，定义决策接口
- BasicAgent: 基于贝叶斯优化的参考实现
- NewAgent: 学生自定义实现模板
- analyze_shot_for_reward: 击球结果评分函数
"""

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


class BayesianBasicAgent(Agent):
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

        print("BasicAgent (Smart, pooltool-native) 已初始化。")

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

            # 1.动态创建“奖励函数” (Wrapper)
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

                # 使用我们的“裁判”来打分
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


class BasicAgent(Agent):
    def __init__(self, target_balls=None, n_simulations=50, c_puct=1.414):
        super().__init__()
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.ball_radius = 0.028575
        self.sim_noise = {"V0": 0.1, "phi": 0.15, "theta": 0.1, "a": 0.005, "b": 0.005}

    def _calc_angle_degrees(self, v):
        angle = math.degrees(math.atan2(v[1], v[0]))
        return angle % 360

    def _get_ghost_ball_target(self, cue_pos, obj_pos, pocket_pos):
        vec_obj_to_pocket = np.array(pocket_pos) - np.array(obj_pos)
        dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)
        if dist_obj_to_pocket == 0:
            return 0.0, 0.0
        unit_vec = vec_obj_to_pocket / dist_obj_to_pocket
        ghost_pos = np.array(obj_pos) - unit_vec * (2 * self.ball_radius)
        vec_cue_to_ghost = ghost_pos - np.array(cue_pos)
        dist_cue_to_ghost = np.linalg.norm(vec_cue_to_ghost)
        phi = self._calc_angle_degrees(vec_cue_to_ghost)
        return float(phi), float(dist_cue_to_ghost)

    def _iter_pocket_positions(self, table):
        if not hasattr(table, "pockets") or table.pockets is None:
            return []

        pockets = table.pockets
        pocket_objs = pockets.values() if isinstance(pockets, dict) else pockets

        pocket_positions = []
        for pocket in pocket_objs:
            if hasattr(pocket, "center"):
                pocket_positions.append(pocket.center)
            elif hasattr(pocket, "position"):
                pocket_positions.append(pocket.position)
            elif isinstance(pocket, (list, tuple, np.ndarray)):
                pocket_positions.append(np.array(pocket))
        return pocket_positions

    def generate_heuristic_actions(self, balls, my_targets, table):
        actions = []

        cue_ball = balls.get("cue") if balls is not None else None
        if cue_ball is None:
            return [self._random_action()]

        if my_targets is None:
            return [self._random_action()]

        cue_pos = cue_ball.state.rvw[0]
        pocket_positions = self._iter_pocket_positions(table)
        if not pocket_positions:
            return [self._random_action()]

        target_ids = []
        for bid in my_targets:
            if bid in balls and balls[bid].state.s != 4:
                target_ids.append(bid)

        if not target_ids:
            if "8" in balls and balls["8"].state.s != 4:
                target_ids = ["8"]
            else:
                return [self._random_action()]

        for tid in target_ids:
            obj_ball = balls[tid]
            obj_pos = obj_ball.state.rvw[0]

            for pocket_pos in pocket_positions:
                phi_ideal, dist = self._get_ghost_ball_target(
                    cue_pos, obj_pos, pocket_pos
                )

                v_base = 1.5 + dist * 1.5
                v_base = float(np.clip(v_base, 1.0, 6.0))

                actions.append(
                    {"V0": v_base, "phi": phi_ideal, "theta": 0, "a": 0, "b": 0}
                )
                actions.append(
                    {
                        "V0": float(min(v_base + 1.5, 7.5)),
                        "phi": phi_ideal,
                        "theta": 0,
                        "a": 0,
                        "b": 0,
                    }
                )
                actions.append(
                    {
                        "V0": v_base,
                        "phi": (phi_ideal + 0.5) % 360,
                        "theta": 0,
                        "a": 0,
                        "b": 0,
                    }
                )
                actions.append(
                    {
                        "V0": v_base,
                        "phi": (phi_ideal - 0.5) % 360,
                        "theta": 0,
                        "a": 0,
                        "b": 0,
                    }
                )

        if len(actions) == 0:
            for _ in range(5):
                actions.append(self._random_action())

        random.shuffle(actions)
        return actions[:30]

    def simulate_action(self, balls, table, action):
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

        try:
            noisy_V0 = np.clip(
                action["V0"] + np.random.normal(0, self.sim_noise["V0"]), 0.5, 8.0
            )
            noisy_phi = (
                action["phi"] + np.random.normal(0, self.sim_noise["phi"])
            ) % 360
            noisy_theta = np.clip(
                action["theta"] + np.random.normal(0, self.sim_noise["theta"]), 0, 90
            )
            noisy_a = np.clip(
                action["a"] + np.random.normal(0, self.sim_noise["a"]), -0.5, 0.5
            )
            noisy_b = np.clip(
                action["b"] + np.random.normal(0, self.sim_noise["b"]), -0.5, 0.5
            )

            cue.set_state(
                V0=float(noisy_V0),
                phi=float(noisy_phi),
                theta=float(noisy_theta),
                a=float(noisy_a),
                b=float(noisy_b),
            )

            if not simulate_with_timeout(shot, timeout=3):
                return None
            return shot
        except Exception:
            return None

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None or table is None or my_targets is None:
            return self._random_action()

        remaining = [
            bid for bid in my_targets if bid in balls and balls[bid].state.s != 4
        ]
        if len(remaining) == 0:
            my_targets = ["8"]

        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

        candidate_actions = self.generate_heuristic_actions(balls, my_targets, table)
        n_candidates = len(candidate_actions)
        if n_candidates == 0:
            return self._random_action()

        N = np.zeros(n_candidates, dtype=np.float64)
        Q = np.zeros(n_candidates, dtype=np.float64)

        raw_min = -1600.0
        raw_max = 400.0
        raw_range = raw_max - raw_min

        for i in range(self.n_simulations):
            if i < n_candidates:
                idx = i
            else:
                total_n = float(np.sum(N))
                ucb_values = (Q / (N + 1e-6)) + self.c_puct * np.sqrt(
                    np.log(total_n + 1.0) / (N + 1e-6)
                )
                idx = int(np.argmax(ucb_values))

            shot = self.simulate_action(balls, table, candidate_actions[idx])
            if shot is None:
                raw_reward = raw_min
            else:
                raw_reward = float(
                    analyze_shot_for_reward(shot, last_state_snapshot, my_targets)
                )

            normalized_reward = (raw_reward - raw_min) / raw_range
            normalized_reward = float(np.clip(normalized_reward, 0.0, 1.0))

            N[idx] += 1.0
            Q[idx] += normalized_reward

        avg_rewards = Q / (N + 1e-6)
        best_idx = int(np.argmax(avg_rewards))
        return candidate_actions[best_idx]


# 兼容性别名：以前叫 NewAgent，现在统一叫 BasicAgent
# 这样旧代码（如 poolenv.py）导入 NewAgent 时也能工作
class NewAgent(BasicAgent):
    pass


class ValueNetwork(nn.Module):
    """顶层：价值网络 (评估局面好坏)"""

    def __init__(self, state_dim=64, hidden_dim=256):
        super(ValueNetwork, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)  # 输出标量 Value

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        value = self.v(x)
        return value


class PyramidAgent(Agent):
    """
    三层金字塔架构 Agent

    1. 底层 (Geometry): 几何筛选候选球路
    2. 中层 (Simulation): 带噪声模拟评估进球率
    3. 顶层 (RL): 价值网络评估走位
    """

    def __init__(self, model_path=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 3. 顶层 - 价值网络 (暂未启用)
        self.value_net = ValueNetwork().to(self.device)
        self.target_value_net = ValueNetwork().to(self.device)
        self.target_value_net.load_state_dict(self.value_net.state_dict())

        # 优化器 (用于训练)
        self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=3e-4)

        # if model_path and os.path.exists(model_path):
        #     try:
        #         self.value_net.load_state_dict(
        #             torch.load(model_path, map_location=self.device)
        #         )
        #         print(f"[PyramidAgent] 成功加载价值网络: {model_path}")
        #     except Exception as e:
        #         print(f"[PyramidAgent] 加载模型失败: {e}")

        # 物理参数 (标准台球参数)
        self.BALL_RADIUS = 0.028575

    def _get_vector(self, p1, p2):
        """计算从 p1 指向 p2 的向量 (返回3D向量, z=0)"""
        return np.array([p2[0] - p1[0], p2[1] - p1[1], 0.0])

    def _angle_between(self, v1, v2):
        """计算两个向量的夹角 (度)"""
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0
        cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    def _generate_candidates(self, balls, my_targets, table):
        """
        底层：几何筛选
        遍历所有 (目标球, 袋口) 组合，计算理论进球参数
        """
        candidates = []

        cue_ball = balls.get("cue")
        if not cue_ball:
            return []

        cue_pos = cue_ball.state.rvw[0]

        # 0. 开球检测 (优化)
        # 1. 必须是满球状态 (16个球)
        # 2. 必须没有分球 (my_targets 包含所有球 或 1-7)
        # 3. 关键：球必须聚在一起 (Rack 状态)
        #    计算 1-15 号球位置的标准差

        if len(balls) >= 16:
            ball_positions = []
            for bid, ball in balls.items():
                if bid == "cue":
                    continue
                ball_positions.append(ball.state.rvw[0])

            if ball_positions:
                ball_positions = np.array(ball_positions)
                # 计算位置标准差 (衡量离散程度)
                std_dev = np.std(ball_positions, axis=0)
                # Rack 状态下，球非常密集，std_dev 应该很小
                # 尤其是 x 轴 (假设摆在 x 轴上) 或 y 轴
                # 通常 Rack 的长宽大约是 4-5 个球直径 (0.25m)
                # 如果 std_dev < 0.2，说明球聚在一起

                # 另外，开球时白球通常在开球线后
                # 假设球桌长 L，开球线在 -L/4 或 L/4
                # 这里简单判断紧密度
                mean_std = np.mean(std_dev)

                if mean_std < 0.15:  # 经验阈值，球堆紧密
                    remaining_own = [
                        bid
                        for bid in my_targets
                        if bid in balls and balls[bid].state.s != 4
                    ]
                    candidates_for_break = (
                        remaining_own
                        if len(remaining_own) > 0
                        else (
                            ["8"] if ("8" in balls and balls["8"].state.s != 4) else []
                        )
                    )
                    if not candidates_for_break:
                        candidates_for_break = [
                            bid
                            for bid, ball in balls.items()
                            if bid != "cue" and ball.state.s != 4
                        ]

                    min_dist = float("inf")
                    apex_ball_id = None
                    for bid in candidates_for_break:
                        dist = np.linalg.norm(balls[bid].state.rvw[0] - cue_pos)
                        if dist < min_dist:
                            min_dist = dist
                            apex_ball_id = bid

                    if apex_ball_id:
                        # 生成对着这个球的大力击球
                        apex_pos = balls[apex_ball_id].state.rvw[0]
                        vec = self._get_vector(cue_pos, apex_pos)
                        phi = np.degrees(np.arctan2(vec[1], vec[0])) % 360

                        print(
                            f"[PyramidAgent] 检测到开球局面 (球堆紧密 std={mean_std:.3f})，直接生成开球动作"
                        )
                        return [
                            {
                                "target_id": apex_ball_id,
                                "V0": 6.0,
                                "phi": phi,
                                "theta": 0,
                                "a": 0,
                                "b": 0,
                                "cut_angle": 0,
                                "type": "break",
                            }
                        ]

        # 获取所有袋口位置
        # pooltool 的 table.pockets 通常是一个字典或列表
        pockets = []
        if hasattr(table, "pockets") and isinstance(table.pockets, dict):
            for pocket in table.pockets.values():
                # 尝试获取袋口中心坐标
                if hasattr(pocket, "center"):
                    pockets.append(pocket.center)
                # 兼容旧版本或不同数据结构
                elif hasattr(pocket, "position"):
                    pockets.append(pocket.position)
                elif isinstance(pocket, (list, tuple, np.ndarray)):
                    pockets.append(np.array(pocket))
                else:
                    # 如果无法获取坐标，打印警告但继续
                    # print(f"[PyramidAgent] Warning: Unknown pocket format: {type(pocket)}")
                    pass

        if not pockets:
            print("[PyramidAgent] Warning: Failed to retrieve pocket coordinates!")

        remaining_own = [
            bid for bid in my_targets if bid in balls and balls[bid].state.s != 4
        ]
        if len(remaining_own) == 0:
            targets_to_check = (
                ["8"] if ("8" in balls and balls["8"].state.s != 4) else []
            )
        else:
            targets_to_check = remaining_own

        for target_id in targets_to_check:
            if target_id not in balls:
                continue

            target_ball = balls[target_id]

            # 关键修复：如果目标球已经进袋 (s=4)，跳过！
            if target_ball.state.s == 4:
                continue

            target_pos = target_ball.state.rvw[0]

            for pocket_pos in pockets:
                # 1. 计算 Ghost Ball 位置 (瞄准点)
                # 向量: 目标球 -> 袋口
                vec_target_pocket = self._get_vector(target_pos, pocket_pos)
                dist_tp = np.linalg.norm(vec_target_pocket)
                if dist_tp == 0:
                    continue

                # 单位向量
                unit_tp = vec_target_pocket / dist_tp

                # Ghost Ball 位置 = 目标球位置 - 2*R * unit_vector
                # (要在接触瞬间，白球中心位于目标球中心后方 2R 处)
                ghost_pos = target_pos - (
                    2 * self.BALL_RADIUS * np.array([unit_tp[0], unit_tp[1], 0])
                )

                # 2. 计算白球击打角度 (phi)
                vec_cue_ghost = self._get_vector(cue_pos, ghost_pos)
                dist_cg = np.linalg.norm(vec_cue_ghost)

                if dist_cg == 0:
                    continue

                phi = np.degrees(np.arctan2(vec_cue_ghost[1], vec_cue_ghost[0])) % 360

                # 3. 检查切球角度 (Cut Angle)
                # 向量: 白球 -> 目标球
                vec_cue_target = self._get_vector(cue_pos, target_pos)
                cut_angle = self._angle_between(vec_cue_target, vec_target_pocket)

                # 如果切角过大 (例如 > 80度)，极难进球，过滤掉
                if cut_angle > 80:
                    continue

                # 4. 阻挡检测
                # 检查白球到 Ghost Ball 的路径上是否有其他球
                # 简化算法：计算其他球到线段 (cue_pos, ghost_pos) 的垂直距离
                # 如果距离 < 2*R，则认为有阻挡
                # 修正：Ghost Ball 位置是假想的白球中心位置。
                # 实际上我们需要检查的是：白球沿路径移动时扫过的圆柱体区域。
                # 半径应该是 2*R (白球半径+障碍球半径)，因为我们要避免两球边缘接触。

                is_obstructed = False
                for other_bid, other_ball in balls.items():
                    if other_bid in ["cue", target_id]:
                        continue

                    # 排除已经进袋的球
                    if other_ball.state.s == 4:
                        continue

                    other_pos = other_ball.state.rvw[0]

                    # 向量: 白球 -> 障碍球
                    vec_cue_obs = self._get_vector(cue_pos, other_pos)
                    # 向量: 白球 -> Ghost Ball (瞄准线)
                    # vec_cue_ghost 已经在上面计算过

                    # 投影长度 (scalar)
                    proj_len = np.dot(vec_cue_obs, vec_cue_ghost) / dist_cg

                    # 如果投影在路径范围内 (0 < proj < dist_cg)
                    if 0 < proj_len < dist_cg:
                        # 垂直距离
                        dist_perp = np.linalg.norm(
                            vec_cue_obs - proj_len * (vec_cue_ghost / dist_cg)
                        )
                        # 严格判定：如果垂直距离 < 2R，说明球心距离小于2R，会发生碰撞
                        if (
                            dist_perp < 2 * self.BALL_RADIUS * 0.98
                        ):  # 稍微留一点容错 margin
                            is_obstructed = True
                            break

                # 检查目标球进袋路径是否受阻 (Target Ball -> Pocket)
                # 这一步同样重要，否则会打到别的球
                if not is_obstructed:
                    vec_target_pocket = self._get_vector(target_pos, pocket_pos)
                    dist_tp = np.linalg.norm(vec_target_pocket)

                    for other_bid, other_ball in balls.items():
                        if other_bid in ["cue", target_id]:
                            continue
                        # 排除已经进袋的球
                        if other_ball.state.s == 4:
                            continue

                        other_pos = other_ball.state.rvw[0]

                        vec_target_obs = self._get_vector(target_pos, other_pos)
                        proj_len = np.dot(vec_target_obs, vec_target_pocket) / dist_tp

                        if 0 < proj_len < dist_tp:
                            dist_perp = np.linalg.norm(
                                vec_target_obs
                                - proj_len * (vec_target_pocket / dist_tp)
                            )
                            if dist_perp < 2 * self.BALL_RADIUS * 0.98:
                                is_obstructed = True
                                break

                if not is_obstructed:
                    # 5. 添加直球候选
                    for v0 in [0.8, 1.0, 1.4, 2.0, 2.5, 3.0]:
                        candidates.append(
                            {
                                "target_id": target_id,
                                "V0": v0,
                                "phi": phi,
                                "theta": 0,
                                "a": 0,
                                "b": 0,
                                "cut_angle": cut_angle,
                                "type": "direct",  # 标记类型
                            }
                        )

                # 6. (新增) 翻袋/解球逻辑 (Kick Shot / Bank Shot)
                # 仅当直球被阻挡，或者这是黑8且很难打时尝试
                # 简化实现：只考虑白球撞一次库边打目标球 (Kick Shot)
                # 镜像法：
                # 1. 将目标点 (Ghost Ball) 关于库边镜像
                # 2. 连接白球和镜像点，交点即为撞库点
                # 3. 检查路径是否通畅

                # 仅在必要时计算 (比如直球受阻，或者为了增加选择)
                # 如果直球不可行，或者这是关键球，则尝试解球
                if is_obstructed or target_id == "8":
                    # 库边定义 (假设标准台面中心在 (W/2, L/2))
                    # x 范围 [0, W], y 范围 [0, L]
                    # 需要从 poolenv 或 table 对象获取准确的边界
                    # 假设 table 对象有 center 和 dimensions
                    # pooltool 默认原点在球桌中心，还是角落？
                    # 观察 poolenv.py:
                    # width = table.w (0.99)
                    # length = table.l (1.98)
                    # 按照惯例，PoolTool 的原点 (0,0) 通常在球桌长方形的中心。
                    # 那么边界是 x: [-W/2, W/2], y: [-L/2, L/2]

                    W = table.w
                    L = table.l
                    rails = [
                        {
                            "name": "left",
                            "axis": 0,
                            "val": -W / 2 + self.BALL_RADIUS,
                        },  # 左库 (x = -W/2)
                        {
                            "name": "right",
                            "axis": 0,
                            "val": W / 2 - self.BALL_RADIUS,
                        },  # 右库 (x = W/2)
                        {
                            "name": "bottom",
                            "axis": 1,
                            "val": -L / 2 + self.BALL_RADIUS,
                        },  # 下库 (y = -L/2)
                        {
                            "name": "top",
                            "axis": 1,
                            "val": L / 2 - self.BALL_RADIUS,
                        },  # 上库 (y = L/2)
                    ]

                    for rail in rails:
                        # 1. 镜像 Ghost Ball
                        ghost_mirror = ghost_pos.copy()
                        ghost_mirror[rail["axis"]] = (
                            2 * rail["val"] - ghost_pos[rail["axis"]]
                        )

                        # 2. 计算白球到镜像点的角度
                        vec_cue_mirror = self._get_vector(cue_pos, ghost_mirror)
                        dist_cm = np.linalg.norm(vec_cue_mirror)
                        if dist_cm == 0:
                            continue

                        phi_kick = (
                            np.degrees(np.arctan2(vec_cue_mirror[1], vec_cue_mirror[0]))
                            % 360
                        )

                        # 3. 计算撞库点 (用于阻挡检测)
                        # 线段: cue_pos -> ghost_mirror
                        # 与库边 rail['val'] 的交点
                        # P = cue + t * (mirror - cue)
                        # P[axis] = rail['val']
                        # t = (rail['val'] - cue[axis]) / (mirror[axis] - cue[axis])

                        if (
                            abs(ghost_mirror[rail["axis"]] - cue_pos[rail["axis"]])
                            < 1e-6
                        ):
                            continue  # 平行

                        t = (rail["val"] - cue_pos[rail["axis"]]) / (
                            ghost_mirror[rail["axis"]] - cue_pos[rail["axis"]]
                        )

                        if not (0 < t < 1):
                            continue  # 撞库点不在两球之间（背向撞库？暂不考虑）

                        impact_point = cue_pos + t * (ghost_mirror - cue_pos)

                        # 4. 检查撞库点是否在台面上 (不能撞出界)
                        other_axis = 1 - rail["axis"]
                        limit = (L / 2 if other_axis == 1 else W / 2) - self.BALL_RADIUS
                        if abs(impact_point[other_axis]) > limit:
                            continue  # 撞在袋口区域外或非法区域

                        # 5. 阻挡检测 (两段路径)
                        # Path 1: Cue -> Impact
                        path1_blocked = False
                        vec_cue_imp = self._get_vector(cue_pos, impact_point)
                        dist_ci = np.linalg.norm(vec_cue_imp)

                        for other_bid, other_ball in balls.items():
                            if other_bid in ["cue", target_id]:
                                continue
                            if other_ball.state.s == 4:  # 忽略已进袋的球
                                continue
                            other_pos = other_ball.state.rvw[0]
                            vec_cue_obs = self._get_vector(cue_pos, other_pos)
                            proj = np.dot(vec_cue_obs, vec_cue_imp) / dist_ci
                            if 0 < proj < dist_ci:
                                dist_perp = np.linalg.norm(
                                    vec_cue_obs - proj * (vec_cue_imp / dist_ci)
                                )
                                if dist_perp < 2 * self.BALL_RADIUS * 0.98:
                                    path1_blocked = True
                                    break
                        if path1_blocked:
                            continue

                        # Path 2: Impact -> Ghost
                        path2_blocked = False
                        vec_imp_ghost = self._get_vector(impact_point, ghost_pos)
                        dist_ig = np.linalg.norm(vec_imp_ghost)

                        for other_bid, other_ball in balls.items():
                            if other_bid in ["cue", target_id]:
                                continue
                            if other_ball.state.s == 4:  # 忽略已进袋的球
                                continue
                            other_pos = other_ball.state.rvw[0]
                            vec_imp_obs = self._get_vector(impact_point, other_pos)
                            proj = np.dot(vec_imp_obs, vec_imp_ghost) / dist_ig
                            if 0 < proj < dist_ig:
                                dist_perp = np.linalg.norm(
                                    vec_imp_obs - proj * (vec_imp_ghost / dist_ig)
                                )
                                if dist_perp < 2 * self.BALL_RADIUS * 0.98:
                                    path2_blocked = True
                                    break
                        if path2_blocked:
                            continue

                        # 6. 添加解球候选
                        # 解球通常需要稍大的力度
                        for v0 in [2.2, 2.6, 3.0]:
                            candidates.append(
                                {
                                    "target_id": target_id,
                                    "V0": v0,
                                    "phi": phi_kick,
                                    "theta": 0,
                                    "a": 0,  # 简化，不加塞
                                    "b": 0,
                                    "cut_angle": cut_angle,  # 近似
                                    "type": "kick",  # 标记为解球
                                }
                            )

                    # 7. (新增) 翻袋逻辑 (Bank Shot)
                    # 母球 -> 目标球 -> 库边 -> 袋口
                    # 场景：目标球进袋的直线被挡，或者角度不好，把目标球撞向库边反弹进袋

                    for rail in rails:
                        # 1. 计算 Ghost Pocket (袋口关于库边的镜像)
                        # 注意：翻袋是目标球反弹，所以是袋口镜像
                        pocket_mirror = pocket_pos.copy()
                        pocket_mirror[rail["axis"]] = (
                            2 * rail["val"] - pocket_pos[rail["axis"]]
                        )

                        # 2. 计算目标球撞库点 (Bank Point)
                        # 连线: 目标球 -> Ghost Pocket
                        vec_target_mirror = self._get_vector(target_pos, pocket_mirror)
                        dist_tm = np.linalg.norm(vec_target_mirror)
                        if dist_tm == 0:
                            continue

                        # 计算 Bank Point
                        # B = T + t * (P_mirror - T)
                        # B[axis] = rail_val
                        if abs(vec_target_mirror[rail["axis"]]) < 1e-6:
                            continue  # 平行

                        t = (
                            rail["val"] - target_pos[rail["axis"]]
                        ) / vec_target_mirror[rail["axis"]]

                        # t 必须 > 0 (朝向库边)，通常 < 1 (但在镜像空间里，Target到Mirror的距离肯定跨过Rail)
                        # 实际上，Bank Point 必须在 Target 和 Pocket_Mirror 之间，所以 0 < t < 1
                        if not (0 < t < 1):
                            continue

                        bank_point = target_pos + t * vec_target_mirror.flatten()

                        # 3. 检查 Bank Point 是否有效 (在台面上)
                        # 注意：target_pos 和 vec_target_mirror 通常是 shape (3,) 的 numpy array
                        # 如果 vec_target_mirror 是 (3,1) 或其他形状，会导致 broadcasting 错误
                        # 确保它们都是 1D array

                        # other_axis = 1 - rail["axis"]
                        # limit = (L / 2 if other_axis == 1 else W / 2) - self.BALL_RADIUS
                        # if abs(bank_point[other_axis]) > limit:
                        #     continue

                        # 重新实现更稳健的坐标检查
                        # rail["axis"] 是 0 (x) 或 1 (y)
                        # 如果是左右库 (axis=0)，我们需要检查 y (axis=1) 是否出界
                        # 如果是上下库 (axis=1)，我们需要检查 x (axis=0) 是否出界

                        check_axis = 1 - rail["axis"]
                        limit_val = (
                            L / 2 if check_axis == 1 else W / 2
                        ) - self.BALL_RADIUS

                        if abs(bank_point[check_axis]) > limit_val:
                            continue

                        # 4. 计算 Ghost Ball (白球瞄准点)
                        # 目标球要沿着 (Target -> Bank Point) 的方向运动
                        vec_target_bank = self._get_vector(target_pos, bank_point)
                        dist_tb = np.linalg.norm(vec_target_bank)
                        if dist_tb == 0:
                            continue

                        unit_tb = vec_target_bank / dist_tb

                        # Ghost Ball 在 Target Ball 后方 2R 处
                        ghost_pos_bank = target_pos - 2 * self.BALL_RADIUS * np.array(
                            [unit_tb[0], unit_tb[1], 0]
                        )

                        # 5. 计算白球击打参数
                        vec_cue_ghost_bank = self._get_vector(cue_pos, ghost_pos_bank)
                        dist_cgb = np.linalg.norm(vec_cue_ghost_bank)
                        if dist_cgb == 0:
                            continue

                        phi_bank = (
                            np.degrees(
                                np.arctan2(vec_cue_ghost_bank[1], vec_cue_ghost_bank[0])
                            )
                            % 360
                        )

                        # 计算切角 (白球 -> 目标球 vs 目标球 -> 库边)
                        vec_cue_target = self._get_vector(cue_pos, target_pos)
                        cut_angle_bank = self._angle_between(
                            vec_cue_target, vec_target_bank
                        )

                        if cut_angle_bank > 80:  # 切角过大
                            continue

                        # 6. 阻挡检测 (三段路径)
                        # Path 1: Cue -> Ghost Ball (Bank)
                        path1_blocked = False
                        vec_cue_gb = self._get_vector(cue_pos, ghost_pos_bank)  # 近似
                        dist_cgb = np.linalg.norm(vec_cue_gb)

                        for other_bid, other_ball in balls.items():
                            if other_bid in ["cue", target_id]:
                                continue
                            if other_ball.state.s == 4:
                                continue

                            other_pos = other_ball.state.rvw[0]
                            vec_cue_obs = self._get_vector(cue_pos, other_pos)
                            proj = np.dot(vec_cue_obs, vec_cue_gb) / dist_cgb

                            if 0 < proj < dist_cgb:
                                dist_perp = np.linalg.norm(
                                    vec_cue_obs - proj * (vec_cue_gb / dist_cgb)
                                )
                                if dist_perp < 2 * self.BALL_RADIUS * 0.98:
                                    path1_blocked = True
                                    break
                        if path1_blocked:
                            continue

                        # Path 2: Target -> Bank Point
                        path2_blocked = False
                        # vec_target_bank 已计算
                        for other_bid, other_ball in balls.items():
                            if other_bid in ["cue", target_id]:
                                continue
                            if other_ball.state.s == 4:
                                continue

                            other_pos = other_ball.state.rvw[0]
                            vec_target_obs = self._get_vector(target_pos, other_pos)
                            proj = np.dot(vec_target_obs, vec_target_bank) / dist_tb

                            if 0 < proj < dist_tb:
                                dist_perp = np.linalg.norm(
                                    vec_target_obs - proj * (vec_target_bank / dist_tb)
                                )
                                if dist_perp < 2 * self.BALL_RADIUS * 0.98:
                                    path2_blocked = True
                                    break
                        if path2_blocked:
                            continue

                        # Path 3: Bank Point -> Pocket
                        path3_blocked = False
                        vec_bank_pocket = self._get_vector(bank_point, pocket_pos)
                        dist_bp = np.linalg.norm(vec_bank_pocket)

                        for other_bid, other_ball in balls.items():
                            if other_bid in ["cue", target_id]:
                                continue
                            if other_ball.state.s == 4:
                                continue

                            other_pos = other_ball.state.rvw[0]
                            vec_bank_obs = self._get_vector(bank_point, other_pos)
                            proj = np.dot(vec_bank_obs, vec_bank_pocket) / dist_bp

                            if 0 < proj < dist_bp:
                                dist_perp = np.linalg.norm(
                                    vec_bank_obs - proj * (vec_bank_pocket / dist_bp)
                                )
                                if dist_perp < 2 * self.BALL_RADIUS * 0.98:
                                    path3_blocked = True
                                    break
                        if path3_blocked:
                            continue

                        # 7. 添加翻袋候选
                        # 翻袋需要较大力度
                        for v0 in [2.6, 2.8, 3.0]:
                            candidates.append(
                                {
                                    "target_id": target_id,
                                    "V0": v0,
                                    "phi": phi_bank,
                                    "theta": 0,
                                    "a": 0,
                                    "b": 0,
                                    "cut_angle": cut_angle_bank,
                                    "type": "bank",  # 标记为翻袋
                                }
                            )

        return candidates

    def _simulate_shot(self, candidate, table, balls, n_sims=None):
        """
        中层：带噪声模拟
        返回: (进球概率, 击球后平均状态向量)
        """
        # 复制环境
        # 注意：为了效率，这里可能需要优化。完整deepcopy比较慢。
        # 增加模拟次数以提高稳定性
        N_SIMS = 10
        success_count = 0
        final_states = []

        # 噪声参数 (模拟执行误差)
        NOISE_PHI = 0.2  # 度
        NOISE_V0 = 0.05  # m/s

        scratch_count = 0  # 白球进袋计数
        foul_count = 0

        # 针对黑8或关键球增加模拟次数
        is_critical = False
        if candidate["target_id"] == "8":
            is_critical = True

        current_n_sims = (
            n_sims if n_sims is not None else (20 if is_critical else N_SIMS)
        )
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

        for _ in range(current_n_sims):
            sim_table = copy.deepcopy(table)
            sim_balls = {bid: copy.deepcopy(b) for bid, b in balls.items()}
            cue = pt.Cue(cue_ball_id="cue")
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

            # 施加噪声
            actual_phi = candidate["phi"] + np.random.normal(0, NOISE_PHI)
            actual_v0 = candidate["V0"] + np.random.normal(0, NOISE_V0)
            actual_v0 = np.clip(actual_v0, 0.5, 8.0)

            shot.cue.set_state(
                V0=actual_v0,
                phi=actual_phi,
                theta=candidate["theta"],
                a=candidate["a"],
                b=candidate["b"],
            )

            # 运行物理引擎
            pt.simulate(shot, inplace=True)

            target_id = candidate["target_id"]
            is_success = False
            events = shot.events
            first_contact_ball_id = None
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
                    if (
                        first_contact_ball_id is not None
                        and first_contact_ball_id in ids
                    ):
                        target_hit_cushion = True
            new_pocketed = [
                bid
                for bid, b in shot.balls.items()
                if b.state.s == 4 and sim_balls[bid].state.s != 4
            ]
            foul_first_hit = (
                first_contact_ball_id is None or first_contact_ball_id != target_id
            )
            foul_no_rail = (
                len(new_pocketed) == 0
                and first_contact_ball_id is not None
                and (not cue_hit_cushion)
                and (not target_hit_cushion)
            )
            if foul_first_hit or foul_no_rail:
                foul_count += 1

            # 严重惩罚：白球进袋
            if shot.balls["cue"].state.s == 4:
                # 如果白球进了，这不仅不是成功，还是大事故
                # 我们在统计 prob 时只统计成功次数，所以这里不算成功
                # 但我们需要传递这个“危险信息”给上层
                # 为了简化，我们在 simulate 阶段不直接处理惩罚，而是通过 prob 体现
                # 或者，我们可以让 prob 变成负数？不行，prob 必须是概率。
                # 方案：在 simulate 返回值中增加一个 risk_factor (自杀率)
                pass
            elif (
                (not foul_first_hit)
                and target_id in shot.balls
                and shot.balls[target_id].state.s == 4
            ):
                is_success = True

            if is_success:
                success_count += 1

            # 统计白球进袋次数
            if shot.balls["cue"].state.s == 4:
                scratch_count += 1

            # 记录状态用于计算 Value
            final_states.append(shot.balls)

        prob = success_count / current_n_sims
        scratch_prob = scratch_count / current_n_sims  # 白球进袋概率
        foul_prob = foul_count / current_n_sims

        avg_state_dict = final_states[0]  # 简化

        return prob, scratch_prob, foul_prob, avg_state_dict

    def process_state(self, balls, my_targets, table):
        """复用 NewAgent 的状态处理"""
        L = table.l
        W = table.w
        remaining_own = [
            bid for bid in my_targets if bid in balls and balls[bid].state.s != 4
        ]
        if len(remaining_own) == 0:
            target_set = {"8"} if ("8" in balls and balls["8"].state.s != 4) else set()
        else:
            target_set = set(remaining_own)
        features = []
        ball_ids = ["cue"] + [str(i) for i in range(1, 16)]
        for bid in ball_ids:
            if bid in balls:
                ball = balls[bid]
                pos = ball.state.rvw[0]
                x = pos[0] / L
                y = pos[1] / W
                is_target = 1.0 if bid in target_set else 0.0
                is_pocketed = 1.0 if ball.state.s == 4 else 0.0
                features.extend([x, y, is_target, is_pocketed])
            else:
                features.extend([0, 0, 0, 1.0])
        return torch.FloatTensor(features).to(self.device)

    def decision(self, balls=None, my_targets=None, table=None):
        """
        顶层决策
        """
        if balls is None:
            return self._random_action()

        # 1. 底层：生成候选
        candidates = self._generate_candidates(balls, my_targets, table)

        if not candidates:
            # 智能保底策略 (Smart Fallback)
            # 当找不到进球路线时，尝试瞄准任意一个有效目标球，力求不犯规
            print("[PyramidAgent] 无可行进球路线，启动智能保底策略...")
            cue_pos = balls["cue"].state.rvw[0]

            fallback_candidates = []

            # 遍历所有有效目标球
            remaining_own = [
                bid for bid in my_targets if bid in balls and balls[bid].state.s != 4
            ]
            if len(remaining_own) == 0:
                targets_to_check = (
                    ["8"] if ("8" in balls and balls["8"].state.s != 4) else []
                )
            else:
                targets_to_check = remaining_own

            for target_id in targets_to_check:
                if target_id not in balls or balls[target_id].state.s == 4:
                    continue
                target_pos = balls[target_id].state.rvw[0]

                # 1. 尝试直接瞄准 (Full hit)
                vec = self._get_vector(cue_pos, target_pos)
                dist = np.linalg.norm(vec)
                if dist > 0:
                    phi = np.degrees(np.arctan2(vec[1], vec[0])) % 360
                    # 添加几个不同力度的直球
                    for v0 in [0.8, 1.5, 3.0]:
                        fallback_candidates.append(
                            {
                                "target_id": target_id,
                                "V0": v0,
                                "phi": phi,
                                "theta": 0,
                                "a": 0,
                                "b": 0,
                                "cut_angle": 0,
                                "type": "fallback_direct",
                            }
                        )

                # 2. 尝试撞库解球 (Kick Shot) 以碰到目标球
                # 场景：直球被挡，尝试通过撞库绕过障碍
                W = table.w
                L = table.l
                rails = [
                    {"name": "left", "axis": 0, "val": -W / 2 + self.BALL_RADIUS},
                    {"name": "right", "axis": 0, "val": W / 2 - self.BALL_RADIUS},
                    {"name": "bottom", "axis": 1, "val": -L / 2 + self.BALL_RADIUS},
                    {"name": "top", "axis": 1, "val": L / 2 - self.BALL_RADIUS},
                ]

                for rail in rails:
                    # 目标球关于库边的镜像点
                    target_mirror = target_pos.copy()
                    target_mirror[rail["axis"]] = (
                        2 * rail["val"] - target_pos[rail["axis"]]
                    )

                    # 白球 -> 镜像点
                    vec_cue_mirror = self._get_vector(cue_pos, target_mirror)
                    dist_cm = np.linalg.norm(vec_cue_mirror)
                    if dist_cm == 0:
                        continue

                    phi_kick = (
                        np.degrees(np.arctan2(vec_cue_mirror[1], vec_cue_mirror[0]))
                        % 360
                    )

                    # 检查撞库点是否在台面上
                    if abs(vec_cue_mirror[rail["axis"]]) < 1e-6:
                        continue
                    t = (rail["val"] - cue_pos[rail["axis"]]) / vec_cue_mirror[
                        rail["axis"]
                    ]
                    if not (0 < t < 1):
                        continue

                    impact_point = (
                        cue_pos + t * vec_cue_mirror.flatten()
                    )  # 确保维度匹配

                    check_axis = 1 - rail["axis"]
                    limit_val = (L / 2 if check_axis == 1 else W / 2) - self.BALL_RADIUS
                    if abs(impact_point[check_axis]) > limit_val:
                        continue

                    # 不做严格的阻挡检测（因为是保底策略，试一试总比没有好）
                    # 但力度要稍大一点，因为路程变长了
                    for v0 in [2.5, 3.0]:
                        fallback_candidates.append(
                            {
                                "target_id": target_id,
                                "V0": v0,
                                "phi": phi_kick,
                                "theta": 0,
                                "a": 0,
                                "b": 0,
                                "cut_angle": 0,
                                "type": "fallback_kick",
                            }
                        )

            if fallback_candidates:
                candidates = fallback_candidates
                print(
                    f"[PyramidAgent] 生成了 {len(candidates)} 个保底候选 (只求碰到球)"
                )
            else:
                print(
                    "[PyramidAgent] 连保底球路都找不到 (可能被完全包围)，只能随机乱打"
                )
                return self._random_action()

        best_score = -float("inf")
        best_action = candidates[0]

        # 2. 中层 & 顶层：模拟评估
        print(f"[PyramidAgent] 评估 {len(candidates)} 个候选球路...")
        remaining_own_before = [
            bid for bid in my_targets if bid in balls and balls[bid].state.s != 4
        ]

        scratch_penalty = 250.0
        foul_penalty = 150.0
        quick_n_sims = 8 if len(remaining_own_before) == 0 else 4
        scored = []
        for cand in candidates:
            prob, scratch_prob, foul_prob, _ = self._simulate_shot(
                cand, table, balls, n_sims=quick_n_sims
            )
            if cand["target_id"] == "8":
                success_reward = 300.0 if len(remaining_own_before) == 0 else -900.0
            else:
                success_reward = 100.0
            rough_ev = (
                prob * success_reward
                + (1 - prob) * (-0.5)
                - scratch_penalty * scratch_prob
                - foul_penalty * foul_prob
            )
            scored.append((rough_ev, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = min(12, len(scored))
        eval_candidates = [cand for _, cand in scored[:top_k]]

        for cand in eval_candidates:
            prob, scratch_prob, foul_prob, final_balls = self._simulate_shot(
                cand, table, balls
            )
            state_tensor = self.process_state(final_balls, my_targets, table).unsqueeze(
                0
            )
            with torch.no_grad():
                position_value = self.value_net(state_tensor).item()

            if cand["target_id"] == "8":
                success_reward = 300.0 if len(remaining_own_before) == 0 else -900.0
            else:
                success_reward = 100.0

            expected_value = (
                prob * (success_reward + 0.99 * position_value)
                + (1 - prob) * (-0.5)
                - scratch_penalty * scratch_prob
                - foul_penalty * foul_prob
            )

            if expected_value > best_score:
                best_score = expected_value
                best_action = cand

        print(
            f"[PyramidAgent] 最佳选择: 目标{best_action['target_id']}, "
            f"角度{best_action['phi']:.1f}, 综合评分(EV){best_score:.2f}"
        )

        return best_action
