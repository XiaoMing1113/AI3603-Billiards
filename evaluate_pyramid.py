import pooltool as pt
from poolenv import PoolEnv
from agent import BasicAgent, NewAgent, PyramidAgent
import time
import numpy as np


def _create_viewer():
    if hasattr(pt, "ShotViewer"):
        return pt.ShotViewer()
    if hasattr(pt, "ani") and hasattr(pt.ani, "ShotViewer"):
        return pt.ani.ShotViewer()
    return None


def _render_shot(viewer, system, title):
    if hasattr(pt, "show"):
        return pt.show(system, title=title)
    if viewer is not None and hasattr(viewer, "show"):
        return viewer.show(system, title=title)
    raise AttributeError(
        "pooltool missing render API (no pt.show and no ShotViewer). "
        "Try installing visualization deps: pip install panda3d simplepbr"
    )

def evaluate():
    # 1. 初始化环境
    env = PoolEnv()
    viewer = _create_viewer()
    
    # 2. 初始化 Agent
    # Player A 使用金字塔 Agent
    agent_a = PyramidAgent(model_path="value_net.pth")
    # Player B 使用基础 Agent
    agent_b = BasicAgent()
    
    n_games = 5
    wins = [0, 0] # A, B
    
    for i_game in range(n_games):
        print(f"\n=== 第 {i_game + 1} 局 ===")
        # 重置环境，指定 target_ball='solid' (A打实心)
        env.reset(target_ball='solid')
        
        turn_count = 0
        while True:
            turn_count += 1
            player = env.get_curr_player() # 'A' or 'B'
            
            # 获取观测
            balls, my_targets, table = env.get_observation(player)
            
            print(f"第 {turn_count} 回合: 玩家 {player} (剩余目标: {len(my_targets)}) 思考中...")
            
            # Agent 决策
            if player == 'A':
                action = agent_a.decision(balls=balls, my_targets=my_targets, table=table)
            else:
                action = agent_b.decision(balls=balls, my_targets=my_targets, table=table)
            
            # 执行击球
            env.take_shot(action)
            
            # 渲染当前杆
            # 注意：PoolEnv.shot_record 存储了 System 对象
            try:
                _render_shot(
                    viewer,
                    env.shot_record[-1],
                    title=f"第 {i_game+1} 局 - 第 {turn_count} 杆 (玩家 {player})",
                )
            except Exception as e:
                print(f"渲染失败: {e}")
            
            # 检查游戏是否结束
            done, info = env.get_done()
            if done:
                winner = info['winner']
                print(f"第 {i_game + 1} 局结束. 获胜者: {winner}")
                if winner == 'A':
                    wins[0] += 1
                elif winner == 'B':
                    wins[1] += 1
                break
                
    print(f"\n最终比分: 玩家 A {wins[0]} - {wins[1]} 玩家 B")

if __name__ == "__main__":
    evaluate()
