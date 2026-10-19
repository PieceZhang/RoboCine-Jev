固定AGILE v1.2、Isaac Lab v2.3.2与本机Sim5.1的完整声明依赖安装在Robot_G1/.venv，venv只读继承原conda。通过config/constraints.txt保留已验收的torch2.7.0、numpy1.26.0、tensordict0.8.3、matplotlib3.10.3。pin2.7/cmeel-boost1.83满足NumPy1.x的兼容性；mujoco固定3.3.7。

上游元数据自身有四处冲突，本地补丁只改安装版本声明：

1. IsaacLab Starlette0.49.1与Sim5.1固定FastAPI不兼容，改为0.45.3。
2. AGILE protobuf3.20.3与IsaacLab>=4.25.8及wandb不兼容，改为>=4.25.8,<7，并约束到6.33.6。
3. AGILE matplotlib3.6.0改为已装且实测使用的3.10.3。
4. AGILE datasets2.19.0要求fsspec≤2024.3.1，而Sim5.1固定fsspec2024.6.1；改用datasets2.21.0，并固定pyarrow16.1.0避免新版移除其旧接口。

补丁见patches，安装脚本幂等应用并执行pip check。不修改模型、物理、电机、观测、策略或控制频率。

Sim5.1扩展内置Warp1.8.2，但PyPI没有1.8.2发行版，故PyPI声明依赖固定到最近的1.8.1。补全依赖后已通过results/final_env_smoke_01的4秒物理smoke，确认Kit实际加载的仍为Sim内置Warp1.8.2，路径记录在metadata.json。200个物理控制周期的全部JSON状态与原smoke_02完全一致，完整65秒验收证据与原环境快照单独保留。未执行学习训练、MuJoCo跨仿真或实机部署验收；安装这些开发依赖不代表训练任务已经全部实测。

完整依赖安装后pip check返回“No broken requirements found.”，结果见results/pip_check.txt；环境快照见config/environment.freeze.txt。

最终smoke的Kit正常退出（child_returncode=0，未触发监督器终止），STOP回调问题已修复。
