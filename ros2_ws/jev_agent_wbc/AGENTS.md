# Closed-loop video delivery

User preference, persisted 2026-10-16: every delivered CPU whole-body closed-loop
video and PhysX whole-body closed-loop video must show the unified Representation.
Use the default `scripts/render_agent_demos.py` or `scripts/render_tpr_demo.py`
pipeline and the shared `jev_agent_wbc/representation_video.py` display.

Reserve a readable area for both modules' conservative occupancy envelopes,
`T_world_mount` and `T_mount_camera` as full 6D transforms, and concise time,
source and spatial relationship text. Follow recorded geometry granularity.
For PhysX, use measured physical robot state, full base rotation and base height;
do not replace robot geometry with a virtual stabilized camera pose.
Only use observations available at the displayed time. Missing/stale spatial
evidence is unknown. Track-center distances are not obstacle surface clearances.
Pure floating-camera experiments have no two-part robot to visualize.

Keep continuous simulation-time playback without explanatory freezes. Preserve
existing recordings and source logs; use a new output directory for rerecordings.
Raw RGB sensor streams are measurement artifacts; final films must pass through
the Representation-enabled renderer. Save provenance and decode-check the film.
Keep recording changes in the presentation layer and preserve concurrent edits
to execution, planning, control and PhysX compatibility code.

PhysX preference, updated 2026-10-17: use the virtual stabilized camera as the
default Agent camera feedback, retaining measured physical robot state. Rerun
the closed loop when changing this source. Delivered PhysX films show measured
overview, a detached camera following the original reference in the same current
scene/time, and `增稳相机（Agent输入）`; omit raw execution RGB from the film.
Keep raw measurements for audit. Acceleration excess is diagnostic only for
PhysX acceptance; network-only timeout trials are unverified, not failures.
