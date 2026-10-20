"""Keep causal film decisions bound to their successful logged API requests."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('agent_demo_requests', ROOT/'scripts/render_agent_demos.py')
RENDERER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDERER)


@pytest.mark.parametrize('questions', [('strategy', 'configuration', 'candidate'),
                                     ('generator', 'construction', 'candidate')])
@pytest.mark.parametrize('retry', [False, True])
def test_stage_matching_preserves_causality_and_ignores_rejected_replies(tmp_path, questions, retry):
    scene = json.loads((ROOT/'scenarios/orbit/runtime_scene.json').read_text())
    shutil.copy2(ROOT/'scenarios/orbit/task.json', tmp_path/'original_task.json')
    (tmp_path/'runtime_scene.json').write_text(json.dumps(scene))
    (tmp_path/'summary.json').write_text(json.dumps(dict(backend='kinematic_integration', mode='jev')))
    reference = RENDERER.OriginalReference(tmp_path/'original_task.json', scene['subject']['position'])
    rows = [dict(t=t, camera_position=reference.sample(t)[0].tolist(),
                 info=dict(target_position=reference.sample(t)[0].tolist())) for t in (0., 4.)]
    (tmp_path/'telemetry.json').write_text(json.dumps(rows))

    stages = ['strategy', 'configuration', 'selection']
    requests = ['strategy_requested', 'search_configuration_requested', 'selection_requested']
    terminals = ['search_strategy', 'search_configuration', 'candidate_approved']
    events, api = [], []
    for i, (stage, question, request, terminal) in enumerate(zip(stages, questions, requests, terminals)):
        evidence = dict(stage=stage, time_s=i+.5)
        events += [dict(event=request, time_s=i+.5, evidence=evidence),
                   dict(event=terminal, time_s=i+1.)]
        record = dict(request=dict(questions={question: {}}, state=json.dumps(evidence)),
                      response=dict(model='test-model', answers={question: dict(choice='logged')}))
        if retry:
            api.append(dict(record, error='rejected protocol reply', status='invalid_protocol_reply'))
        api.append(record)
    (tmp_path/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    (tmp_path/'jev_api.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in api))

    run = RENDERER.Run(tmp_path, representation_panel=False)
    assert list(run.request_at) == [0, 2, 4]
    assert list(run.api_at) == [1, 3, 5]
    assert all('error' not in a for a in run.api_at.values())
    assert run.state(.75)['answer'] is None
    assert next(iter(run.state(1.)['answer']['request']['questions'])) == questions[0]
    assert next(iter(run.state(2.)['answer']['request']['questions'])) == questions[1]
    assert next(iter(run.state(3.)['answer']['request']['questions'])) == questions[2]
