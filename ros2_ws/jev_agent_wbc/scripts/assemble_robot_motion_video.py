#!/usr/bin/env python3
"""Audit physical-state captures, concatenate three 1x clips, decode-check MP4."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    import numpy as np
    from scipy.spatial.transform import Rotation
    audits = []
    chapters = [";FFMETADATA1", "title=三种机器人运动检查", "comment=Measured PhysX link-state replay at original speed; no avoidance Agent"]
    inputs = []
    offset_ms = 0
    for robot, title in [("g1","Unitree G1 / AGILE WBC"), ("go2","Go2 + Piper / LeggedManip WBC"),
                          ("ridgeback","Ridgeback + UR5 / planar-base WBC")]:
        folder = root / robot
        rendered = root / (robot + "_video")
        meta = json.loads((folder / "capture.json").read_text())
        replay = json.loads((rendered / "replay_audit.json").read_text())
        assert replay["valid"] and replay["physics_steps_during_render"] == 0
        data = np.load(folder / "measured_links.npz", allow_pickle=False)
        poses, times = data["poses"], data["times"]
        frames = read_rows(folder / "frames.jsonl")
        if robot == "g1":
            rows = read_rows(folder / "physical/trajectory.jsonl")
            summary = json.loads((folder / "physical/summary.json").read_text())
            assert summary["stable_multidirection_pass"]
            base_positions = np.array([r["base_position"] for r in rows])
            base_id = 0
        elif robot == "go2":
            rows = read_rows(folder / "physical/measurements.jsonl")
            summary = json.loads((folder / "physical/summary.json").read_text())
            assert summary["all_pass"]
            base_positions = np.array([r["base_pos"] for r in rows])
            base_id = meta["body_names"].index("base")
        else:
            rows = read_rows(folder / "physical/samples.jsonl")
            summary = json.loads((folder / "physical/acceptance.json").read_text())
            assert summary["passed"]
            base_positions = np.array([r["base_p"] for r in rows])
            base_id = meta["body_names"].index("base_link")
        assert len(rows) == len(frames) == len(poses)
        position_error = float(np.max(np.linalg.norm(poses[:,base_id,:3]-base_positions, axis=-1)))
        assert position_error < 1e-6
        assert sha(folder / "validator_source.py") == meta["source_sha256"]
        timestamp_error = max(abs((frame.get("case_time") if robot=="go2" else frame["time"])-
                                  row.get("time",row.get("t"))) for frame,row in zip(frames,rows))
        assert timestamp_error < 1e-8
        ee_position_error = ee_rotation_error = None
        if robot != "g1":
            ee_name = "end_effector" if robot == "go2" else "ur_arm_wrist_3_link"
            ee_id = meta["body_names"].index(ee_name)
            ee_pose = np.array([r["ee_world_pose_xyz_wxyz"] if robot=="go2" else r["ee_p"]+r["ee_qwxyz"] for r in rows])
            ee_position_error = float(np.max(np.linalg.norm(poses[:,ee_id,:3]-ee_pose[:,:3],axis=-1)))
            actual = Rotation.from_quat(poses[:,ee_id][:,[4,5,6,3]])
            logged = Rotation.from_quat(ee_pose[:,[4,5,6,3]])
            ee_rotation_error = float(np.max((actual*logged.inv()).magnitude()))
            assert ee_position_error < 1e-6 and ee_rotation_error < 1e-6
        video = rendered / (robot + ".mp4")
        probe = json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0",
            "-show_entries","stream=width,height,nb_frames,r_frame_rate:format=duration","-of","json",str(video)],text=True))
        stream = probe["streams"][0]
        assert int(stream["nb_frames"]) == replay["frame_count"]
        assert stream["width"] == 1280 and stream["height"] == 720 and stream["r_frame_rate"] == "25/1"
        duration_ms = round(float(probe["format"]["duration"]) * 1000)
        assert abs(duration_ms - round(float(times[-1])*1000)) <= 1
        inputs.append("file '" + str(video) + "'")
        chapters += ["[CHAPTER]","TIMEBASE=1/1000",f"START={offset_ms}",f"END={offset_ms+duration_ms}","title="+title]
        audits.append(dict(robot=robot,physical_validation_passed=True,physical_rows=len(rows),
                           captured_links=len(meta["body_paths"]),duration_s=duration_ms/1000,
                           max_base_position_difference_m=position_error,max_timestamp_difference_s=timestamp_error,
                           max_ee_position_difference_m=ee_position_error,max_ee_rotation_difference_rad=ee_rotation_error,
                           capture_manifest_sha256=sha(folder/"capture.json"),
                           measured_links_sha256=sha(folder/"measured_links.npz"),replay=replay,
                           video_sha256=sha(video),chapter_start_s=offset_ms/1000))
        offset_ms += duration_ms
    (root / "concat.txt").write_text("\n".join(inputs)+"\n")
    (root / "chapters.txt").write_text("\n".join(chapters)+"\n")
    final = root / "three_robots_motion.mp4"
    if final.exists():
        raise FileExistsError(final)
    subprocess.run(["ffmpeg","-v","error","-f","concat","-safe","0","-i",str(root/"concat.txt"),
        "-f","ffmetadata","-i",str(root/"chapters.txt"),"-map","0:v:0","-map_metadata","1","-map_chapters","1",
        "-c","copy","-movflags","+faststart",str(final)],check=True)
    subprocess.run(["ffmpeg","-v","error","-xerror","-i",str(final),"-map","0:v:0","-f","null","-"],check=True)
    manifest = dict(artifact=final.name,sha256=sha(final),duration_s=offset_ms/1000,width=1280,height=720,
                    fps=25,decode_check_passed=True,measured_state_replay=True,playback_speed=1,
                    agent_connected=False,robots=audits,
                    scope="Controller inspection video, measured per-link physical poses redrawn in Isaac Sim. Decorative grid and XY traces are presentation only.")
    (root/"VIDEO_PROVENANCE.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (root/"README.txt").write_text(
        "三种机器人运动检查视频\n\n"
        "three_robots_motion.mp4：1280×720、25fps、216秒，原速1×。\n"
        "00:00–01:05 Unitree G1 / AGILE：前后左右、斜向、转向及停止。\n"
        "01:05–02:25 Go2 + Piper：十个独立工况，含行走与机械臂同时运动。工况之间重置。\n"
        "02:25–03:36 Ridgeback + UR5：全向移动、转向及20秒全身协调。\n\n"
        "本片为Isaac Sim重新运行控制器后采集的实测连杆状态回放，不是直播RGB。\n"
        "所有连杆位姿均从PhysX读取，按原始时间播放；渲染不积分物理、不执行策略。\n"
        "逐帧核对回放世界变换与实测位姿，并用原始控制器日志独立核对基座、末端和时间。\n"
        "VIDEO_PROVENANCE.json含视频SHA256、全片解码检查及各平台审计。\n"
        "每个机器人目录的physical/保留本轮运行日志、控制检查及原始测量。\n\n"
        "范围：没有接入避障Agent或相机任务轨迹。Go2使用Piper，不是D1。\n"
        "Ridgeback为官方平面关节底盘，不能用本片评价麦克纳姆轮地面摩擦。\n"
        "新增网格、轨迹和字幕为检查展示，不参与控制或碰撞检测。\n")
    print(json.dumps({"video":str(final),"duration_s":offset_ms/1000,"bytes":final.stat().st_size,"audit_passed":True},indent=2))


if __name__ == "__main__":
    main()
