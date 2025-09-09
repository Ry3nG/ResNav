# 2025/09/09 Map Fix

## 问题回顾
- **原始问题**: seed 20021213生成不可行地图，导致训练卡住
- **根本原因**: pallet放置算法没有考虑全局路径约束

## 修复方案
**最小化修复**: 在现有`amr_env/sim/scenario_manager.py`中添加路径可行性检查和重试机制

### 修改内容
1. **添加路径可行性检查函数** (`_is_path_feasible`)
   - 使用BFS算法检查起点到终点是否有可行路径
   - 仅30行代码，轻量级实现

2. **添加重试机制** (在`sample`方法中)
   - 最多重试5次生成可行scenario
   - 每次重试使用新的随机状态
   - 保持原有接口不变

### 代码变更
```python
# 在 scenario_manager.py 中添加:
def _is_path_feasible(self, grid, start_pose, goal_xy, resolution):
    # BFS路径检查逻辑

def sample(self):
    # 原有逻辑 + 重试机制
    for attempt in range(max_retries):
        grid, waypoints, start_pose, goal_xy, info = create_blockage_scenario(cfg, self._rng)
        if self._is_path_feasible(grid, start_pose, goal_xy, cfg.resolution_m):
            return grid, waypoints, start_pose, goal_xy, info
        self._rng = np.random.default_rng(self._rng.integers(0, 2**32))
```

## 测试结果
✅ **所有测试通过**:
- seed 20021213: 现在生成可行scenario
- 多个问题seed: 6/6 生成可行scenario
- 重试机制: 100% 成功率

## 优势
1. **最小化修改**: 仅修改一个文件，添加50行代码
2. **向后兼容**: 保持原有接口不变
3. **高效**: 轻量级BFS检查，性能影响最小
4. **可靠**: 重试机制确保高成功率
