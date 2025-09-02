# Changelog

All notable changes to the ME5418 AMR Navigation project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

##2025-09-02

### Added

#### Core Framework
- **Environment**: Gymnasium-compatible `UnicycleNavEnv` for AMR navigation simulation
- **Robot Model**: `UnicycleModel` wrapper around roboticstoolbox-python's Unicycle class
- **Sensor System**: `LidarSensor` for 2D sparse laser range finding (24-36 beams, 270° FOV)
- **Constants**: Centralized configuration system with environment, robot, and sensor parameters

#### Maps & Path Generation
- **S-Path Generator**: `make_s_map()` for creating sinusoidal test trajectories
- **Map Infrastructure**: Base classes and utilities for procedural map generation

#### Controllers (Baselines)
- **Pure Pursuit + APF**: Classical path-following with artificial potential fields
- **Trap-Aware PP+APF**: Enhanced variant with temporal decision-making and stuck detection
- Both controllers support:
  - Configurable lookahead distances and gains
  - Obstacle repulsion with distance-based scaling
  - Goal attraction and path following
  - Anti-oscillation mechanisms

#### Visualization & Analysis
- **Real-time Visualization**: Pygame-based renderer with:
  - Robot pose and trajectory tracking
  - LiDAR beam visualization
  - Obstacle and path overlay
  - Performance metrics display
- **Post-hoc Analysis**: Matplotlib utilities for trajectory and performance plotting

#### Reward System
- **Navigation Rewards**: Multi-component reward function including:
  - Path progress rewards
  - Collision penalties with exponential distance scaling
  - Lateral and heading deviation penalties
  - Velocity smoothness incentives
  - Goal reaching bonuses

#### Scripts & Utilities
- **Baseline Evaluation**: `baseline_eval.py` for systematic controller assessment
- **Manual Control**: Human teleoperation interface for data collection
- **Episode Recording**: Utilities for capturing and storing simulation episodes
- **Random Agent**: Baseline random action policy

#### Development Infrastructure
- **Environment Setup**: Conda environment configuration with all dependencies
- **Package Structure**: Modular, extensible codebase organization
- **Git Integration**: Proper .gitignore and repository structure

### Technical Specifications
- **Python Version**: 3.10+ compatibility
- **Key Dependencies**:
  - Gymnasium (RL environment interface)
  - roboticstoolbox-python (kinematic models)
  - Stable-Baselines3 (future RL integration)
  - Pygame (visualization)
  - NumPy/SciPy (numerical computing)
- **Robot Parameters**:
  - Differential drive kinematics
  - 0.5m diameter, 1.5 m/s max speed
  - 10 Hz control frequency
- **Sensor Configuration**:
  - 24-beam LiDAR, 4m range
  - 270° field of view
  - 2cm ray marching resolution

### Project Status
- ✅ Core simulation framework functional
- ✅ Baseline controllers implemented and tested
- ✅ Visualization and evaluation tools operational
- 🚧 RL training pipeline in development
- 🚧 Dynamic obstacle scenarios planned
- 🚧 Comprehensive evaluation suite pending

### Known Issues
- Fixed import errors in maps module initialization
- Resolved Unicycle model state setting compatibility issues
- All baseline evaluation scripts now functional

---

**Initial Release**: This represents the foundational codebase for the ME5418 project on learning yield/pass decisions for AMR navigation. The framework is operational with working baseline controllers, ready for RL algorithm integration and extended scenario development.
