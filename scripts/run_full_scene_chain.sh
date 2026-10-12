#!/usr/bin/env bash
# run_full_scene_chain.sh — 全公寓场景（--full 产出）端到端一键链
#
#   前置：datasets/replica_v1/frl_apartment_0/habitat/{mesh_semantic.ply,
#         info_semantic.json} 存在（replica_autosem 的输入；缺失则跳过 ⓪
#         并直接使用仓库里已提交的 frl_apartment_0_autosem_full.json）
#
#   用法:
#     scripts/run_full_scene_chain.sh                     # 默认 result1（全周环绕）
#     scripts/run_full_scene_chain.sh 4                   # 指定 mock 编号（result4 = 推拉变焦）
#     scripts/run_full_scene_chain.sh 1 open              # 跑完自动打开 viewer
#
#   注意:
#     ⓪ 会覆盖已提交的 frl_apartment_0_autosem[_full].json（git 可恢复）
#     build 的环绕可行域钳制可能合法报错（全场景隔断错位，见 README 回归测试节）
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

MOCK_NUM="${1:-1}"
OPEN_VIEWER="${2:-}"
RAW="datasets/CameraMovementsRecommand_output_json/result${MOCK_NUM}.json"
SCENE="cinema_traj/scenes/frl_apartment_0_autosem_full.json"
OUT="cinema_traj_out/frl_apartment_0_autosem_full"

PY="python"; [[ -x .venv/bin/python ]] && PY=".venv/bin/python"

echo "════════════ 全公寓场景一键链（result${MOCK_NUM} → ${SCENE}）════════════"

# ⓪ 场景再生成（依赖 Replica v1 语义网格；缺失则跳过并提示）
MESH="datasets/replica_v1/frl_apartment_0/habitat/mesh_semantic.ply"
if [[ -f "$MESH" ]]; then
    echo "── [0/7] replica_autosem --full（重生成场景）"
    $PY scripts/replica_autosem.py --full
else
    echo "── [0/7] 跳过 replica_autosem（缺 ${MESH}）"
    echo "        使用仓库已提交的 ${SCENE}"
fi

# ① recommend → ② build → ③ optimize → ④ export → ⑤ retime → ⑥ view
echo "── [1/7] recommend（解析 result${MOCK_NUM} → 原子计划）"
$PY run_traj.py --stage recommend --raw "$RAW" --scene "$SCENE" --force

echo "── [2/7] build（anchor + 参数化运镜；环绕会被可行域钳制或报错）"
$PY run_traj.py --stage build --scene "$SCENE" --force

echo "── [3/7] optimize（两阶段 SDF 优化 + 硬门终验）"
$PY run_traj.py --stage optimize --scene "$SCENE" --force

echo "── [4/7] export（通用 JSON/CSV + 无人机设定点）"
$PY run_traj.py --stage export --scene "$SCENE"

echo "── [5/7] retime（能力约束重定时 → 可执行设定点）"
$PY run_traj.py --stage retime --scene "$SCENE" --force

echo "── [6/7] view（自包含 3D 查看器；B 键切云密度，P 点云 / L 标签）"
$PY run_traj.py --stage view --scene "$SCENE"

echo "── [7/7] 完成"
HTML="${OUT}/viewer/frl_apartment_0_autosem_full.html"
echo "viewer: $HTML"
echo "exports: ${OUT}/exports/"
if [[ "$OPEN_VIEWER" == "open" ]]; then
    open "$HTML"
fi
