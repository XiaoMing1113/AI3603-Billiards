import pooltool as pt
from poolenv import PoolEnv
from agent import BasicAgent, NewAgent, PyramidAgent
import time
import numpy as np
import argparse
import copy


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


def _extract_first_contact_ball_id(shot):
    events = getattr(shot, "events", [])
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
        et = str(getattr(e, "event_type", "")).lower()
        ids = list(e.ids) if hasattr(e, "ids") else []
        if ("cushion" not in et) and ("pocket" not in et) and ("cue" in ids):
            other_ids = [i for i in ids if i != "cue" and i in valid_ball_ids]
            if other_ids:
                return other_ids[0]
    return None


def _compute_break_spread(initial_balls, final_balls):
    init_positions = []
    for bid, ball in initial_balls.items():
        if bid == "cue":
            continue
        if getattr(ball.state, "s", None) == 4:
            continue
        init_positions.append(ball.state.rvw[0][:2])
    if not init_positions:
        return 0.0
    rack_center = np.mean(np.array(init_positions, dtype=np.float64), axis=0)

    dists = []
    pocketed = 0
    for bid, ball in final_balls.items():
        if bid == "cue":
            continue
        if getattr(ball.state, "s", None) == 4:
            pocketed += 1
            continue
        pos = ball.state.rvw[0][:2]
        dists.append(float(np.linalg.norm(pos - rack_center)))
    if not dists:
        return 0.0

    spread = float(np.mean(dists))
    spread += 0.03 * float(pocketed)
    return spread


def search_best_break(
    target_ball="solid",
    phi_span=12.0,
    n_phi=81,
    v0_span=1.0,
    n_v0=5,
    repeats=5,
):
    env = PoolEnv()
    env.reset(target_ball=target_ball)
    balls, my_targets, table = env.get_observation("A")

    agent = PyramidAgent(model_path="value_net.pth")
    candidates = agent._generate_candidates(balls, my_targets, table)
    break_cand = None
    for cand in candidates:
        if cand.get("type") in ("break_fast", "break"):
            break_cand = cand
            break
    if break_cand is None:
        break_cand = {"V0": 5.0, "phi": 90.0, "theta": 0.0, "a": 0.0, "b": 0.0}

    base_phi = float(break_cand.get("phi", 90.0))
    base_v0 = float(break_cand.get("V0", 5.0))

    phi_values = np.linspace(base_phi - phi_span, base_phi + phi_span, int(n_phi))
    v0_values = np.linspace(base_v0 - v0_span, base_v0 + v0_span, int(n_v0))
    v0_values = np.clip(v0_values, 2.0, 6.5)

    noise_std = getattr(env, "noise_std", None) or {
        "V0": 0.1,
        "phi": 0.1,
        "theta": 0.1,
        "a": 0.003,
        "b": 0.003,
    }

    results = []
    initial_balls_snapshot = {bid: copy.deepcopy(b) for bid, b in balls.items()}

    start_t = time.perf_counter()
    total_configs = int(n_phi) * int(n_v0)
    done_configs = 0

    for v0 in v0_values:
        for phi in phi_values:
            score_sum = 0.0
            scratch_cnt = 0
            illegal_first_hit_cnt = 0
            pocketed_sum = 0

            for _ in range(int(repeats)):
                sim_table = copy.deepcopy(table)
                sim_balls = {bid: copy.deepcopy(b) for bid, b in balls.items()}
                cue = pt.Cue(cue_ball_id="cue")
                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

                noisy_v0 = float(
                    np.clip(v0 + np.random.normal(0, noise_std["V0"]), 0.5, 8.0)
                )
                noisy_phi = float((phi + np.random.normal(0, noise_std["phi"])) % 360.0)
                noisy_theta = float(
                    np.clip(0.0 + np.random.normal(0, noise_std["theta"]), 0.0, 90.0)
                )
                noisy_a = float(
                    np.clip(0.0 + np.random.normal(0, noise_std["a"]), -0.5, 0.5)
                )
                noisy_b = float(
                    np.clip(0.0 + np.random.normal(0, noise_std["b"]), -0.5, 0.5)
                )
                cue.set_state(
                    V0=noisy_v0, phi=noisy_phi, theta=noisy_theta, a=noisy_a, b=noisy_b
                )

                pt.simulate(shot, inplace=True)

                final_balls = shot.balls
                score = _compute_break_spread(initial_balls_snapshot, final_balls)

                if final_balls["cue"].state.s == 4:
                    scratch_cnt += 1
                    score -= 5.0

                first_contact = _extract_first_contact_ball_id(shot)
                if first_contact is None or (first_contact not in my_targets):
                    illegal_first_hit_cnt += 1
                    score -= 2.0

                pocketed = 0
                for bid, ball in final_balls.items():
                    if bid != "cue" and ball.state.s == 4:
                        pocketed += 1
                pocketed_sum += pocketed

                score_sum += score

            avg_score = score_sum / float(repeats)
            results.append(
                {
                    "avg_score": float(avg_score),
                    "phi": float(phi % 360.0),
                    "v0": float(v0),
                    "dphi": float(((phi - base_phi + 180.0) % 360.0) - 180.0),
                    "scratch_rate": float(scratch_cnt) / float(repeats),
                    "illegal_first_hit_rate": float(illegal_first_hit_cnt)
                    / float(repeats),
                    "avg_pocketed": float(pocketed_sum) / float(repeats),
                }
            )

            done_configs += 1
            if done_configs % 50 == 0 or done_configs == total_configs:
                elapsed = time.perf_counter() - start_t
                print(f"搜索进度: {done_configs}/{total_configs}，用时 {elapsed:.1f}s")

    results.sort(key=lambda x: x["avg_score"], reverse=True)

    print("\n=== 开球搜索结果 (Top 12) ===")
    for i, r in enumerate(results[:12], start=1):
        print(
            f"{i:02d}. score={r['avg_score']:.4f}, V0={r['v0']:.2f}, phi={r['phi']:.2f} "
            f"(dphi={r['dphi']:+.2f}), pocketed={r['avg_pocketed']:.2f}, "
            f"scratch={r['scratch_rate']:.2%}, illegal_first_hit={r['illegal_first_hit_rate']:.2%}"
        )

    best = results[0]
    print(
        f"\n推荐开球: V0={best['v0']:.2f}, phi={best['phi']:.2f} (dphi={best['dphi']:+.2f})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["evaluate", "break_search"], default="evaluate"
    )
    parser.add_argument("--target_ball", choices=["solid", "stripe"], default="solid")
    parser.add_argument("--phi_span", type=float, default=12.0)
    parser.add_argument("--n_phi", type=int, default=81)
    parser.add_argument("--v0_span", type=float, default=1.0)
    parser.add_argument("--n_v0", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if args.mode == "break_search":
        search_best_break(
            target_ball=args.target_ball,
            phi_span=args.phi_span,
            n_phi=args.n_phi,
            v0_span=args.v0_span,
            n_v0=args.n_v0,
            repeats=args.repeats,
        )
    else:
        evaluate()
