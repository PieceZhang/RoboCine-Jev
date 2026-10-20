# Closed-loop video delivery

User preference, updated 2026-10-19: unless explicitly stated otherwise, all Jev
calls must use the official TypeSafe API at `https://api.typesafe.ai/v1/systemone`
with the API key stored in the repository-root `typesafe_Jev_api_key` file.
Timeouts or connection failures when calling the official API do not count as
failures of the current test; record them as network-related, unverified trials.

User preference, updated 2026-10-17: all new CPU executions use `pace=1`.
Do not add deliberate wall-clock slowdown, including in batch scripts, independent
camera experiments or fixed-camera CPU demos. Keep historical recordings, frozen
source snapshots and logs intact. This does not change the candidate curve's
within-shot timing profile.

User preference, updated later 2026-10-17: new PhysX executions also use `pace=1`.
Never pause physical time to wait for Agent trajectories. Use bounded asynchronous
control and latched timeout/failure braking; late results cannot automatically
resume motion. Report measured wall-clock pacing separately from shot/safety
acceptance. Preserve historical slow recordings and frozen configurations.

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

PhysX preference, updated later 2026-10-19: use validated 100 Hz physics and the raw physical
camera as default Agent feedback, bypassing stabilizer computation. Retain the
unchanged Agent/planning workflow, budgets, sensor coverage and safety gates.
Use short diagnostic prefixes for speed iterations and a complete shortest
existing shot for final verification. Rerun
the closed loop when changing this source. Delivered PhysX films show measured
overview, a detached camera following the original reference in the same current
scene/time, and `原始相机（Agent输入）`. Historical stabilized films stay intact.
Keep raw measurements for audit. Acceleration excess is diagnostic only for
PhysX acceptance; network-only timeout trials are unverified, not failures.

Measure complete real-time runs without live RGB recording. Preserve the full
dual-scanner coverage and original mapping algorithm; timestamp asynchronous
mapping results by acquisition time, bound in-flight work, and latch a brake on
worker failure/backlog. Never relax command, sensor, planning or safety thresholds
to obtain a timing pass. Report timing_pass separately from end-to-end success.
Native read-only ray queries may overlap CPU supervision, but all query jobs must
join before any PhysX scene mutation, rendering or physics advance.
Lower frequencies require physical tracking/constraint comparisons; 100 Hz is
preferred over the tested 50 Hz because it better preserves 200 Hz trajectories.

Use --save-replay to save measured per-link transforms without live RGB work;
render_physx_replay.py may later redraw them without integrating physics or
rerunning Agent decisions. Label such video as measured-state replay, validate
every rendered link pose and its timestamp, retain source hashes, then use the
Representation-enabled render_physx_demos.py pipeline. A replay image is not a
new sensor observation and must never be fed back to an earlier Agent decision.

User preference, updated 2026-10-20: transient decision/module failures should
use bounded waiting, retries and resumable checkpoints instead of immediately
failing the task. This supersedes the earlier blanket latch-on-timeout/backlog
preference. Keep PhysX advancing at pace=1, require fresh observations and fresh
commands before resuming, and retain independent physical safety validation.
The requested comparison covers all twelve authored demos against CPU closed
loop; report incomplete/failed quality checks separately from runtime recovery.
