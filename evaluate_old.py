"""
evaluate.py - Agent 评估脚本

功能：
- 让两个 Agent 进行多局对战
- 统计胜负和得分
- 支持切换先后手和球型分配

使用方式：
1. 调整 n_games 设置对战局数
2. 运行脚本查看结果
"""

# 导入必要的模块
import time

from utils import set_random_seed
from poolenv import PoolEnv
from agent import BasicAgent, PyramidAgent, BayesianBasicAgent

# 设置随机种子，enable=True 时使用固定种子，enable=False 时使用完全随机
# 根据需求，我们在这里统一设置随机种子，确保 agent 双方的全局击球扰动使用相同的随机状态
set_random_seed(enable=True, seed=42)

env = PoolEnv()
n_games = 120

my_agent = PyramidAgent(model_path="value_net.pth")
basic_agent = BayesianBasicAgent()

my_wins = 0
basic_wins = 0
draws = 0
my_foul_losses = 0

total_start_t = time.perf_counter()

for i in range(n_games):
    print()
    game_no = i + 1
    target_ball = "solid" if ((i // 2) % 2 == 0) else "stripe"
    swap_sides = (i // 4) % 2 == 1

    if not swap_sides:
        player_a_agent = my_agent
        player_b_agent = basic_agent
    else:
        player_a_agent = basic_agent
        player_b_agent = my_agent

    env.reset(target_ball=target_ball)
    print(f"------- 第 {game_no} 局比赛开始 -------")
    print(
        f"本局 Player A: {player_a_agent.__class__.__name__}, "
        f"Player B: {player_b_agent.__class__.__name__}, "
        f"目标球型: {target_ball}"
    )
    game_start_t = time.perf_counter()
    while True:
        player = env.get_curr_player()
        print(f"[第{env.hit_count}次击球] player: {player}")
        obs = env.get_observation(player)
        if player == "A":
            action = player_a_agent.decision(*obs)
        else:
            action = player_b_agent.decision(*obs)
        step_info = env.take_shot(action)

        done, info = env.get_done()
        if not done:
            # poolenv中已有打印，无需再输出
            # if step_info.get('FOUL_FIRST_HIT'):
            #     print("本杆判罚：首次接触对方球或黑8，直接交换球权。")
            # if step_info.get('NO_POCKET_NO_RAIL'):
            #     print("本杆判罚：无进球且母球或目标球未碰库，直接交换球权。")
            # if step_info.get('NO_HIT'):
            #     print("本杆判罚：白球未接触任何球，直接交换球权。")
            # if step_info.get('ME_INTO_POCKET'):
            #     print(f"我方球入袋：{step_info['ME_INTO_POCKET']}")
            if step_info.get("ENEMY_INTO_POCKET"):
                print(f"对方球入袋：{step_info['ENEMY_INTO_POCKET']}")
        if done:
            winner = info["winner"]
            game_elapsed_s = time.perf_counter() - game_start_t
            total_elapsed_s = time.perf_counter() - total_start_t

            my_player = "A" if player_a_agent is my_agent else "B"
            my_lost = winner in ("A", "B") and winner != my_player

            foul_loss = False
            foul_reason = None
            if my_lost and player == my_player:
                if step_info.get("WHITE_BALL_INTO_POCKET") and step_info.get(
                    "BLACK_BALL_INTO_POCKET"
                ):
                    foul_loss = True
                    foul_reason = "白球+黑8同杆进袋"
                elif step_info.get("BLACK_BALL_INTO_POCKET"):
                    foul_loss = True
                    foul_reason = "清台前误打黑8"

            if winner == "SAME":
                draws += 1
            elif winner == "A":
                if player_a_agent is my_agent:
                    my_wins += 1
                else:
                    basic_wins += 1
            else:
                if player_b_agent is my_agent:
                    my_wins += 1
                else:
                    basic_wins += 1
            print(
                f"当前战绩: PyramidAgent {my_wins} - {basic_wins} BasicAgent, 平局 {draws}"
            )
            print(
                f"本局用时: {int(game_elapsed_s // 60)}m {int(game_elapsed_s % 60)}s，"
                f"总用时: {int(total_elapsed_s // 60)}m {int(total_elapsed_s % 60)}s"
            )
            if my_lost:
                if foul_loss:
                    my_foul_losses += 1
                    print(f"输因: 我方犯规（{foul_reason}）")
                else:
                    print("输因: 非犯规终局（对手正常胜利或超时判定）")
            print(f"累计因犯规输掉局数: {my_foul_losses}")
            break

print(f"\n最终战绩: PyramidAgent {my_wins} - {basic_wins} BasicAgent, 平局 {draws}")
