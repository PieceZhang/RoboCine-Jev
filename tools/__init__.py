"""Interactive viewers / GT tools (not pipeline stages).

  serve_depth_viewer    — depth-map viewer + GT writeback (snapshot 'predicted')
  serve_validation      — VLM-era GT audit server (snapshot 'vlm_predicted')
  serve_mined_gt        — mined-frame GT cleaning server (snapshot 'depth_predicted')
  make_validation_interface — static HTML validator generator
  gt_server_common      — shared HTTP spine (Range video, run_server)
"""
