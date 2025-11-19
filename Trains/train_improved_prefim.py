import sys
import platform
import os

# 设置UTF-8编码输出（Windows兼容）
if platform.system() == 'Windows':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append('..')
sys.path.append('./')

# 导入改进版参数（如果存在）或使用原始参数
try:
    from Parameters.IMPROVED_RLSF_parameters import *
    print("[OK] Using IMPROVED_RLSF_parameters")
except ImportError:
    from Parameters.PREFIM_parameters import *
    print("[WARNING] Using PREFIM_parameters (fallback)")

# Windows多进程优化：减少并行环境数量避免EOFError
if platform.system() == 'Windows':
    if num_envs > 4:
        print(f"[WARNING] Windows detected, reducing num_envs from {num_envs} to 4")
        num_envs = 4
    if eval_num_envs > 4:
        print(f"[WARNING] Windows detected, reducing eval_num_envs from {eval_num_envs} to 4")
        eval_num_envs = 4
    # 设置默认的改进功能参数（更保守更安全），若参数模块已提供则尊重其值
    try:
        enable_bias_correction
    except NameError:
        enable_bias_correction = True
    try:
        enable_uncertainty_modeling
    except NameError:
        enable_uncertainty_modeling = True
    try:
        enable_improved_labeling
    except NameError:
        enable_improved_labeling = True

    if 'bias_correction_config' not in globals():
        bias_correction_config = {
            'enabled': True,
            'window_size': 1000,
            'initial_delta': 0.1,
            'adaptation_rate': 0.03,
            'min_delta': 0.0,
            'max_delta': 1.0,
            'target_violation_rate': 0.03
        }
    else:
        # 可选：轻度收敛促进（仅在缺省或异常值时调整）
        bias_correction_config['adaptation_rate'] = max(1e-4, float(bias_correction_config.get('adaptation_rate', 0.03)))
        bias_correction_config['target_violation_rate'] = float(bias_correction_config.get('target_violation_rate', 0.03))

    if 'uncertainty_config' not in globals():
        uncertainty_config = {
            'enabled': True,
            'uncertainty_penalty': 0.2,
            'exploration_bonus': 0.02,
            'confidence_threshold': 0.8,
            'method': 'ensemble'
        }
    else:
        uncertainty_config['uncertainty_penalty'] = float(uncertainty_config.get('uncertainty_penalty', 0.2))
        uncertainty_config['exploration_bonus'] = float(uncertainty_config.get('exploration_bonus', 0.02))

    if 'labeling_config' not in globals():
        labeling_config = {
            'enabled': True,
            'confidence_threshold': 0.7,
            'conservative': True
        }


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from Sources.utils import save_frames_as_gif

#------------------------------------------#
def main():
    import wandb
    from Sources.utils import create_folder
    global n_ensemble, segment_length, max_episode_length  # ensure we reference and (optionally) update the imported globals

    strat_schedule = args.strat_schedule
    seed = args.seed

    # 保障不确定性建模最小集成规模（提前于命名与权重路径设置）
    try:
        if 'enable_uncertainty_modeling' in globals() and enable_uncertainty_modeling and n_ensemble < 3:
            print(f"[WARNING] n_ensemble raised from {n_ensemble} to 3 for uncertainty modeling stability")
            n_ensemble = 3
    except Exception as _:
        pass

    # 使用改进版命名
    name = f'IMPROVED-RLSF-{n_ensemble}-{env_name}-{strat_schedule}'
    if(wandb_log):
        # 自动注入WANDB_API_KEY，避免控制台交互
        # if not os.environ.get('WANDB_API_KEY'):
        #     os.environ['WANDB_API_KEY'] = '016182a657ce2a6e2431b4d2caf162819769d7bb'
        wandb.init(name=name, group=f'{env_name}', project='RLSF_Improved')

    global weight_path
    weight_path = './weights'
    weight_path = f'{weight_path}/{env_name}/IMPROVED-RLSF-{strat_schedule}/{seed}'
    create_folder(weight_path)

    if(wandb_log):
        # 1. 将命令行参数转换为字典
        config_dict = vars(args)

        # 2. 将你额外想要记录的改进功能配置，也添加到这个字典里
        #    如果 args 里已经有同名的键，这里的会覆盖它，确保值的统一
        config_dict.update({
            'enable_bias_correction': enable_bias_correction,
            'enable_uncertainty_modeling': enable_uncertainty_modeling,
            'enable_improved_labeling': enable_improved_labeling
        })

        # 3. 只调用一次 update，传入最终合并好的配置字典
        wandb.config.update(config_dict)

    #------------------------------------------#
    import safety_gymnasium
    from Sources.wrapper import CostWrapper, NavClassifierWrapper, VelClassifierWrapper

    if('Safety' in env_name):
        sample_env = safety_gymnasium.make(env_name, render_mode='rgb_array', camera_name='fixedfar')
        wrappers = [CostWrapper]

        env = safety_gymnasium.vector.make(env_id=env_name, num_envs=num_envs, wrappers=wrappers)
        if (eval_num_envs):
            test_env = safety_gymnasium.vector.make(env_id=env_name, num_envs=eval_num_envs, wrappers=wrappers)
        else:
            test_env = None

    elif('Driver' in env_name):
        from Sources.envs.Driver.driver import get_driver
        viz = False
        if('viz' in env_name.lower()):
            viz = True
        if('blocking' in env_name.lower()):
            scenario = 'blocked'
        elif('two' in env_name.lower()):
            scenario = 'twolanes'
        elif('change' in env_name.lower()):
            scenario = 'changing_lane'
        elif('stopping' in env_name.lower()):
            scenario = 'stopping'

        sample_env = get_driver(scenario=scenario, viz_obs=viz)

        env = safety_gymnasium.vector.SafetyAsyncVectorEnv([lambda: get_driver(scenario=scenario, viz_obs=viz, constraint=True) for _ in range(num_envs)])
        if (eval_num_envs):
            test_env = safety_gymnasium.vector.SafetyAsyncVectorEnv([lambda: get_driver(scenario=scenario, viz_obs=viz, constraint=True) for _ in range(eval_num_envs)])
        else:
            test_env = None

    elif('BiasedPendulum' in env_name):
        import gymnasium as gym
        from Sources.wrapper import BiasedPendulumWrapper

        def BiasedPendulum():
            env = gym.make('InvertedPendulum-v4')
            env = BiasedPendulumWrapper(env)
            return env

        sample_env = BiasedPendulum()
        env = safety_gymnasium.vector.SafetyAsyncVectorEnv([BiasedPendulum for _ in range(num_envs)])
        if (eval_num_envs):
            test_env = safety_gymnasium.vector.SafetyAsyncVectorEnv([BiasedPendulum for _ in range(eval_num_envs)])
        else:
            test_env = None

    elif('BlockedSwimmer' in env_name):
        import gymnasium as gym
        from Sources.wrapper import BlockedSwimmerWrapper

        def BlockedSwimmer():
            env = gym.make('Swimmer-v4')
            env = BlockedSwimmerWrapper(env)
            return env

        sample_env = BlockedSwimmer()
        env = safety_gymnasium.vector.SafetyAsyncVectorEnv([BlockedSwimmer for _ in range(num_envs)])
        if (eval_num_envs):
            test_env = safety_gymnasium.vector.SafetyAsyncVectorEnv([BlockedSwimmer for _ in range(eval_num_envs)])
        else:
            test_env = None

    else:
        raise ValueError('Unknown environment')
    #------------------------------------------#
    from Sources.algo.prefim import PREFIM
    from Sources.buffer import Trajectory_Buffer_Continuous, Trajectory_Buffer_Query, Schedule
    from copy import deepcopy
    import threading
    import torch
    import setproctitle
    from torch import nn
    from Sources.density import SimHash
    import numpy as np

    # 导入评估指标工具
    try:
        from Sources.utils.evaluation_metrics import RLSFEvaluationMetrics
        evaluator = RLSFEvaluationMetrics(save_dir=weight_path)
        use_evaluator = True
    except ImportError:
        print("[WARNING] Evaluation metrics not available")
        use_evaluator = False

    # 已在函数开头进行 n_ensemble 的最小值保障（避免命名不一致与局部变量遮蔽）

    #------------------------------------------#
    import time

    def evaluate(algo, env,max_episode_length, t):
        global max_eval_return
        mean_return = 0.0
        mean_cost = 0.0
        failed_case = []
        cost_sum = [0 for _ in range(eval_num_envs)]

        for step in range(num_eval_episodes//eval_num_envs):
            state,_ = env.reset()
            episode_return = 0.0
            episode_cost = 0.0
            start_t = time.time()
            EVAL_TIMEOUT_S = 60  # 增加到 60 秒，避免误触发
            for iter in range(max_episode_length):
                # 每 50 步打印一次进度
                if (iter % 50 == 0):
                    print(f'valid {step+1}/{num_eval_episodes//eval_num_envs}: {iter/max_episode_length*100:.2f}% {iter}/{max_episode_length}', end='\r', flush=True)
                # 超时保护，避免评估长时间阻塞
                if time.time() - start_t > EVAL_TIMEOUT_S:
                    print(f"[WARNING] Evaluation timeout after {EVAL_TIMEOUT_S}s at iter {iter}; early stopping this episode.")
                    break
                action = algo.exploit(state)
                state, reward, cost, done, _, _ = env.step(action)
                episode_return += np.sum(reward)
                episode_cost += np.sum(cost)
                for idx in range(eval_num_envs):
                    cost_sum[idx] += cost[idx]
            for idx in range(eval_num_envs):
                failed_case.append(cost_sum[idx])
                cost_sum[idx] = 0
            # 每轮 episode 结束后补打一行完整进度，便于 GUI 日志解析
            print(f'valid {step+1}/{num_eval_episodes//eval_num_envs}: 100.00% {max_episode_length}/{max_episode_length}')
            mean_return += episode_return
            mean_cost += episode_cost

        mean_return = mean_return/num_eval_episodes
        mean_cost = mean_cost/num_eval_episodes
        tmp_arr = np.asarray(failed_case)

        success_rate = np.sum(tmp_arr<=env_cost_limit)/num_eval_episodes
        value = (mean_return * success_rate)/10
        if (value>max_eval_return):
            max_eval_return = value
            algo.save_models(f'{weight_path}/({value:.3f})-({success_rate:.2f})-({mean_return:.2f})-({mean_cost:.2f})')
        else:
            max_eval_return*=0.999
        print(f'[Eval] R: {mean_return:.2f}, C: {mean_cost:.2f}, '+
            f'SR: {success_rate:.2f}, '
            f'V: {value:.2f}, maxV: {max_eval_return:.2f}')
        if(wandb_log):
            wandb.log({'eval/return':mean_return, 'eval/cost':mean_cost})

        # 返回评估结果供自动调参使用
        eval_result = {
            'mean_return': mean_return,
            'mean_cost': mean_cost,
            'success_rate': success_rate,
            'value': value
        }

        # 记录并打印改进版特有的指标（同时输出到 stdout 供 GUI 实时解析）
        if hasattr(algo, 'bias_corrector'):
            try:
                delta_val = getattr(algo.bias_corrector, 'delta', None)
                bias_val = getattr(algo.bias_corrector, 'bias_estimate', None)
                # if delta_val is not None:
                #     print(f": {float(delta_val):.6f}")
                # if bias_val is not None:
                #     print(f"improved/bias_estimate: {float(bias_val):.6f}")
                # if wandb_log and use_evaluator:
                #     wandb.log({
                #         '': float(delta_val) if delta_val is not None else None,
                #         'improved/bias_estimate': float(bias_val) if bias_val is not None else None
                #     })
            except Exception as _e:
                print(f"[WARNING] Failed to export improved metrics: {_e}")

        return eval_result

    def render(env, algo, t):
        state, _ = env.reset()
        done = False
        truncated = False
        rewards = []
        costs = []
        costs_clfs = []
        frames = []

        i = 0
        while not done and not truncated:
            i += 1
            action = algo.exploit([state])[0]

            pred_cost = torch.sigmoid(algo.clfs[0](torch.tensor(np.array([state]), device=device, dtype=torch.float32), torch.tensor(np.array([action]), device=device, dtype=torch.float32))).detach().cpu().numpy()[0]
            clfs_cost = 1.0 if pred_cost > 0.5 else 0.0

            state, reward, cost, done, truncated, _ = env.step(action)

            costs_clfs.append(clfs_cost)
            rewards.append(reward)
            costs.append(cost)
            frames.append(env.render())

        frames = np.array(frames)

        print(f'Episode length: {i}\n')

        gif_path = f'{weight_path}/step_{t}/'
        create_folder(gif_path)
        save_frames_as_gif(frames, path=gif_path, filename='episode.gif', costs=np.cumsum(costs), clfs_costs=np.cumsum(costs_clfs), rewards=np.cumsum(rewards))
        plt.close()


    def train(env,test_env,algo,eval_algo):
        t = [0 for _ in range(num_envs)]
        eval_thread = None
        state,_ = env.reset()

        # 自动调参状态追踪
        poor_eval_count = 0
        last_eval_result = None

        print('start training with IMPROVED RLSF')
        print(f'Bias Correction: {enable_bias_correction}')
        print(f'Uncertainty Modeling: {enable_uncertainty_modeling}')
        print(f'Improved Labeling: {enable_improved_labeling}')

        for step in range(1,num_training_step//num_envs+1):
            if (step%100==0):
                print(f'train: {step/(num_training_step//num_envs)*100:.2f}% {step}/{num_training_step//num_envs}', end='\r')
                if(wandb_log):
                    wandb.log({'train/step': step/(num_training_step//num_envs)*100})
            state, t = algo.step(env, state, t, step*num_envs)
            if algo.is_update(step*num_envs):
                    eval_return.write(f'{np.mean(algo.return_reward)}\n')
                    eval_return.flush()
                    eval_cost.write(f'{np.mean(algo.return_cost)}\n')
                    eval_cost.flush()
                    algo.update()

            if (step) % (eval_interval//num_envs) == 0 or step==1:
                # 为减少 IO，快速测试模式下不在每次评估点保存权重（仅在指标提升或训练结束时保存）
                if (test_env):
                    if eval_thread is not None:
                        eval_thread.join()
                        # 评估线程结束后，检查是否需要自动调参
                        # if last_eval_result is not None:
                        #     sr = last_eval_result['success_rate']
                        #     cost = last_eval_result['mean_cost']
                        #     if sr < 0.1 and cost > 2 * env_cost_limit:
                        #         poor_eval_count += 1
                        #         print(f"[AUTO-TUNE] Poor performance detected ({poor_eval_count}/3): SR={sr:.2f}, C={cost:.2f}")
                        #     else:
                        #         poor_eval_count = 0  # 重置计数器

                        #     # 连续3次不佳则触发自动调参
                        #     if poor_eval_count >= 3:
                        #         print("\n" + "="*60)
                        #         print("[AUTO-TUNE] Triggering automatic hyperparameter adjustment")
                        #         print("="*60)

                        #         # 调整 uncertainty_penalty
                        #         if hasattr(algo, 'uncertainty_estimator') and hasattr(algo.uncertainty_estimator, 'uncertainty_penalty'):
                        #             old_val = algo.uncertainty_estimator.uncertainty_penalty
                        #             algo.uncertainty_estimator.uncertainty_penalty = min(0.8, old_val * 1.25)
                        #             print(f"  uncertainty_penalty: {old_val:.4f} → {algo.uncertainty_estimator.uncertainty_penalty:.4f}")

                        #         # 调整 exploration_bonus
                        #         if hasattr(algo, 'uncertainty_estimator') and hasattr(algo.uncertainty_estimator, 'exploration_bonus'):
                        #             old_val = algo.uncertainty_estimator.exploration_bonus
                        #             algo.uncertainty_estimator.exploration_bonus = max(0.005, old_val * 0.5)
                        #             print(f"  exploration_bonus: {old_val:.4f} → {algo.uncertainty_estimator.exploration_bonus:.4f}")

                        #         # 调整学习率
                        #         if hasattr(algo, 'optim_actor'):
                        #             for param_group in algo.optim_actor.param_groups:
                        #                 old_lr = param_group['lr']
                        #                 param_group['lr'] = max(1e-5, old_lr * 0.8)
                        #                 print(f"  lr_actor: {old_lr:.6f} → {param_group['lr']:.6f}")

                        #         if hasattr(algo, 'optim_critic'):
                        #             for param_group in algo.optim_critic.param_groups:
                        #                 old_lr = param_group['lr']
                        #                 param_group['lr'] = max(1e-5, old_lr * 0.8)
                        #                 print(f"  lr_critic: {old_lr:.6f} → {param_group['lr']:.6f}")

                        #         if hasattr(algo, 'optim_cost_critic'):
                        #             for param_group in algo.optim_cost_critic.param_groups:
                        #                 old_lr = param_group['lr']
                        #                 param_group['lr'] = max(1e-5, old_lr * 0.8)
                        #                 print(f"  lr_cost_critic: {old_lr:.6f} → {param_group['lr']:.6f}")

                        #         # 调整 adaptation_rate
                        #         if hasattr(algo, 'bias_corrector') and hasattr(algo.bias_corrector, 'adaptation_rate'):
                        #             old_val = algo.bias_corrector.adaptation_rate
                        #             algo.bias_corrector.adaptation_rate = min(0.05, old_val * 1.5)
                        #             print(f"  adaptation_rate: {old_val:.4f} → {algo.bias_corrector.adaptation_rate:.4f}")

                        #         print("="*60 + "\n")
                        #         poor_eval_count = 0  # 重置计数器，避免频繁调整

                    eval_algo.copyNetworksFrom(algo)
                    eval_algo.eval()

                    # 使用包装函数捕获评估结果
                    def evaluate_and_capture():
                        nonlocal last_eval_result
                        last_eval_result = evaluate(eval_algo, test_env, max_episode_length, step)

                    eval_thread = threading.Thread(target=evaluate_and_capture)
                    eval_thread.start()
                # Render if applicable
                if(('Driver' in env_name or 'Carla' in env_name or 'Highway' in env_name) and eval_algo is not None):
                    print('Rendering')
                    render(sample_env, eval_algo, step)
        algo.save_models(f'{weight_path}/s{seed}-finish')
        if(eval_thread is not None):
            eval_thread.join()

    state_shape=sample_env.observation_space.shape
    action_shape=sample_env.action_space.shape
    aug_state_shape = None
    if(aug_state):
        if('Circle' in env_name):
            aug_state_shape = (16,)
        elif('Goal' in env_name):
            aug_state_shape = (16*3,)
        else:
            raise ValueError('Unknown environment for Aug State')
    sample_env.close()

    exp_good_buffer = Trajectory_Buffer_Continuous(
        buffer_size=feedback_buffer_size,
        state_shape=state_shape,
        action_shape=action_shape,
        device='cpu',
        aug_state_shape=aug_state_shape,
        priority=False
    )

    exp_bad_buffer = Trajectory_Buffer_Continuous(
        buffer_size=feedback_buffer_size,
        state_shape=state_shape,
        action_shape=action_shape,
        device='cpu',
        aug_state_shape=aug_state_shape,
        priority=False
    )


    strat = strat_schedule.split('_')[0]
    if(len(strat_schedule.split('_'))>1):
        schedule = strat_schedule.split('_')[1]
        print(f'Strat: {strat} Schedule: {schedule}')
        scheduler = Schedule(n_samples_rollout=buffer_size, total_traj_queries=total_queries, max_episode_length=max_episode_length, schedule=schedule, total_timesteps=num_training_step)
    else:
        scheduler = None

    if(aug_state):
        tmp_query_buffer = Trajectory_Buffer_Query(
            segment_length=segment_length,
            env_cost_limit=env_cost_limit,
            state_shape=aug_state_shape,
            action_shape=action_shape,
            scheduler=scheduler
        )
        _state_shape = aug_state_shape

    else:
        tmp_query_buffer = Trajectory_Buffer_Query(
            segment_length=segment_length,
            env_cost_limit=env_cost_limit,
            state_shape=state_shape,
            action_shape=action_shape,
            scheduler=scheduler
        )
        _state_shape = state_shape


    # 保障 max_episode_length 与 segment_length 的可整除关系，避免算法断言失败
    # 已在函数开头声明 global，所以这里只做数值修正
    try:
        if int(max_episode_length) % int(segment_length) != 0:
            print(f"[WARNING] Adjusting segment_length from {segment_length} to {max_episode_length} to satisfy divisibility")
            segment_length = int(max_episode_length)
    except Exception as _:
        pass

    if(strat=='novel'):
        hash_table = SimHash(k=k, state_shape=_state_shape, device=device, action_shape=action_shape, use_actions=False, feature_state_dims=None)

    else:
        hash_table = None

    setproctitle.setproctitle(f'{env_name}-IMPROVED-RLSF-{seed}')

    print("\n🚀 Initializing IMPROVED RLSF Algorithm...")
    algo = PREFIM(env_name=env_name,exp_good_buffer=exp_good_buffer,exp_bad_buffer=exp_bad_buffer, tmp_query_buffer=tmp_query_buffer,
            state_shape=state_shape, action_shape=action_shape,
            device=device, seed=seed, gamma=gamma,cost_gamma=cost_gamma,buffer_size=buffer_size,
            mix=mix, hidden_units_actor=hidden_units_actor,
            hidden_units_critic=hidden_units_critic,units_clfs=hidden_units_clfs,
            lr_actor=lr_actor,lr_critic=lr_critic,lr_cost_critic=lr_cost_critic,lr_penalty=lr_penalty, epoch_ppo=epoch_ppo,
            epoch_clfs=epoch_clfs,batch_size=batch_size,lr_clfs=lr_clfs,clip_eps=clip_eps, lambd=lambd, coef_ent=coef_ent,
            max_grad_norm=max_grad_norm,reward_factor=reward_factor,max_episode_length=max_episode_length,
            env_cost_limit=env_cost_limit,risk_level=risk_level,num_envs=num_envs,
            start_bad=start_bad, wandb_log=wandb_log, alpha=alpha, clip_dev=clip_dev, segment_length=segment_length, n_ensemble=n_ensemble,
            class_prob=class_prob, aug_state=aug_state, aug_state_shape=aug_state_shape,
            pos_weight=pos_weight, strat=strat, encode_action=encode_action, warm_start_steps=warm_start_steps,
            hash_map=hash_table, over_sample=over_sample, hinge_coeff=hinge_coeff, conv=conv)

    print("[OK] IMPROVED RLSF Algorithm initialized successfully!")
    # 同步改进配置到算法实例，确保 CLI/预设参数真正生效
    try:
        if 'bias_correction_config' in globals() and hasattr(algo, 'bias_corrector'):
            if isinstance(bias_correction_config, dict):
                if 'adaptation_rate' in bias_correction_config:
                    algo.bias_corrector.adaptation_rate = float(bias_correction_config.get('adaptation_rate', algo.bias_corrector.adaptation_rate))
                if 'initial_delta' in bias_correction_config:
                    try:
                        # 若 BiasCorrector 暴露 delta，可直接覆盖初始值
                        setattr(algo.bias_corrector, 'delta', float(bias_correction_config.get('initial_delta')))
                    except Exception:
                        pass
        if 'uncertainty_config' in globals() and hasattr(algo, 'uncertainty_estimator'):
            if isinstance(uncertainty_config, dict):
                if 'uncertainty_penalty' in uncertainty_config:
                    algo.uncertainty_estimator.uncertainty_penalty = float(uncertainty_config.get('uncertainty_penalty', getattr(algo.uncertainty_estimator, 'uncertainty_penalty', 0.2)))
                if 'exploration_bonus' in uncertainty_config:
                    algo.uncertainty_estimator.exploration_bonus = float(uncertainty_config.get('exploration_bonus', getattr(algo.uncertainty_estimator, 'exploration_bonus', 0.02)))
        print(
            f"[SYNC] adaptation_rate={getattr(algo.bias_corrector,'adaptation_rate',None)}, "
            f"initial_delta={getattr(algo.bias_corrector,'delta',None)}, "
            f"uncertainty_penalty={getattr(algo.uncertainty_estimator,'uncertainty_penalty',None)}, "
            f"exploration_bonus={getattr(algo.uncertainty_estimator,'exploration_bonus',None)}"
        )
    except Exception as e:
        print(f"[WARNING] Failed to sync improved configs: {e}")


    if(test_env):
        eval_algo = deepcopy(algo)
    else:
        eval_algo = None

    global eval_return, eval_cost, max_eval_return
    eval_return = open(f'{weight_path}/eval_return.txt', 'w')
    eval_cost = open(f'{weight_path}/eval_cost.txt', 'w')
    max_eval_return = -np.inf

    train(env=env,test_env=test_env,algo=algo,eval_algo=eval_algo)

    eval_return.close()
    eval_cost.close()

    env.close()
    if (test_env):
        test_env.close()

    if wandb_log:
        wandb.finish()

if __name__ == '__main__':
    main()

