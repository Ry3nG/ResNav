# ME5418 项目代码瘦身与重构建议

## 1. 现状概览
- **核心模拟与环境 (`amr_env/`)**：包含轻量级 2D 模拟（`sim/`）、Gym 环境（`gym/`）、奖励计算（`reward.py`）。当前 `ResidualNavEnv` 同时承担场景采样、传感器数据、奖励统计等职责，文件长度超过 400 行，维护困难。
- **控制算法 (`control/`)**：仅含纯跟踪 `pure_pursuit.py`。
- **训练脚本 (`training/`)**：依赖 Hydra + Stable-Baselines3，包含向量化环境工厂、特征抽取器、回调、两个训练入口（PPO/SAC）以及回放脚本。
- **可视化 (`visualization/`)**：pygame 渲染器与视频导出工具。
- **工具 (`tools/`)**：交互式 launcher，用于拼装 Hydra 命令和解析实验配置。
- **配置 (`configs/`)**：Hydra 配置树，覆盖 env/robot/reward/algo/network 等多个组。

## 2. 主要痛点
1. **环境类过于臃肿**：`ResidualNavEnv` 同时处理场景采样、LiDAR、奖励、日志（参见 `amr_env/gym/residual_nav_env.py:31` 及之后），违反单一职责，新同学难以定位问题。
2. **奖励接口依赖隐式字段**：`amr_env/reward.py:26` 中的 `compute_terms` 需要调用方事先在 `reward_cfg` 中塞 `_ctx/_dmin/_true_ranges` 等魔术 key，接口不透明。
3. **工具/脚本逻辑重复**：`tools/launcher.py:63` 与 `training/rollout.py:28` 都实现了运行目录解析、配置加载，修改时必须同步两处代码。
4. **训练流程对第三方耦合重**：`training/train_sac.py:66` 结合 Hydra、WandB、VecNormalize，课堂 Demo 场景下显得复杂，上手门槛高。
5. **缺少共享 util 层**：配置解析、奖励上下文、渲染辅助等散落在脚本里，难以复用。
6. **缺少整体架构文档**：README 未说明模块分工，团队成员需要通读源码才能理解整体流程。

## 3. 重构与瘦身建议
### 3.1 抽离公共基础设施
- 新建 `amr_env/utils/`）用于存放运行目录解析、配置加载、奖励辅助结构等基础函数。
- 将 `detect_run_dir_from_model`、`load_run_config` 等搬到公共模块，训练脚本与工具共用，消除重复代码。

### 3.2 拆分环境职责
- 将环境拆分为若干组件，并在 `ResidualNavEnv` 中通过组合使用：
  - `scenario_service.py`：封装场景采样、网格缓存、EDT 计算。
  - `observation_builder.py`：集中管理 LiDAR 采样、路径上下文、观测打包。
  - `reward_manager.py`：负责奖励状态、日志打包。
- 拆分后 `ResidualNavEnv` 主要保留 Gym API、step/reset 流程，文件长度控制在 200 行以内。

### 3.3 简化奖励接口
- 定义显式的 `RewardContext`/`RewardInputs` 数据类，由奖励子系统维护历史状态（例如 `prev_goal_dist`、LiDAR 历史），环境只需调用 `RewardComputer.step(...)`。
- 将奖励权重、拆解逻辑集中在 `reward_manager.py`，避免环境里手动拼 dict。

### 3.4 精简工具链
- 把 `rollout` 与 launcher 共享的工具函数收纳到新建的公共 util，保持单一来源。

### 3.5 文档与导航
- 在 `docs/` 中补充：
  - `ARCHITECTURE.md`：模块说明、关键依赖关系图。
  - `SIMULATION_FLOW.md`：从场景生成 → 环境 → 奖励 → 训练的流程说明。
- 更新 README，增加“项目结构”“快速开始”章节，指向上述文档。

## 4. 最新进展
- 已创建 `amr_env/utils/` 并整合运行目录、配置读取等公共函数，`tools/launcher.py` 与 `training/rollout.py` 改为复用统一接口。
- `ResidualNavEnv` 拆分为 `ScenarioService`、`ObservationBuilder`、`RewardManager` 三个组件，环境主体缩减为组合逻辑，奖励与观测状态改由专用模块维护。
