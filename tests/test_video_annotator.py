"""video_annotator 单测：词表计数、别名归一、夹紧、extract_json、mock 端到端。"""
from pathlib import Path

import pytest

from video_annotator import taxonomy as T
from video_annotator.annotate import (build_prompt, finalize, need_retry,
                                      sanitize)
from video_annotator.llm import extract_json


# ---------------------------------------------------------------- 词表计数
def test_taxonomy_counts():
    assert len(T.SCENE_MAJOR) == 4
    assert [len(v) for v in T.SCENE_MINOR.values()] == [10, 7, 10, 1]
    assert sum(len(v) for v in T.SCENE_MINOR.values()) == 28
    assert len(T.MOVEMENTS) == 22
    assert len(T.movement_names()) == 22
    assert len(T.SUBJECT_MAJOR) == 4
    assert sum(len(v) for v in T.SUBJECT_MINOR.values()) == 15
    assert len(T.PEITI_SCENE) == 16
    assert len(T.PEITI_PERSON) == 16
    assert len(T.peiti_names()) == 30  # 32 项中「独立」「其它」两组同名，去重后 30
    assert len(T.SCENE_FEATURES) == 4


def test_movement_major_membership():
    # 22 个小类恰好被 5 个大类覆盖（组合运镜为空组）
    grouped = [n for names in T.MOVEMENT_MAJOR.values() for n in names]
    assert sorted(grouped) == sorted(T.movement_names())
    assert T.movement_major_of("前推运镜") == "基础运镜"
    assert T.movement_major_of("希区柯克") == "电影运镜"
    assert T.movement_major_of("百万运镜") == "网感运镜"
    assert T.movement_major_of("静止") == "无"


def test_canonical_aliases():
    assert T.canonical("城市") == "城市-日景"
    assert T.canonical("情侣1") == "情侣"
    assert T.canonical("快速摇摄（含甩镜转场）") == "快速摇摄"
    assert T.canonical("人+宠物") == "人＋宠物"
    assert T.canonical("跟随") == "主角跟随"
    assert T.canonical("公路/铁路/桥") == "公路、铁路、桥"
    # 词表内原样返回
    assert T.canonical("咖啡厅、花店") == "咖啡厅、花店"
    assert T.canonical("楼梯台阶") == "楼梯台阶"


def test_peiti_group():
    assert T.peiti_group("雕塑") == "场景培体"
    assert T.peiti_group("围巾") == "人物培体"
    assert T.peiti_group("独立") == "场景培体_人物培体"
    assert T.peiti_group("不存在") == ""


# ---------------------------------------------------------------- extract_json
def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    text = "好的，结果如下：\n```json\n{\"场景大类\": \"室内\"}\n```\n以上。"
    assert extract_json(text) == {"场景大类": "室内"}


def test_extract_json_prose_and_nested_braces():
    text = '标注结果 {"a": {"b": "包含}大括号"}, "c": [1,2]} 请查收'
    assert extract_json(text) == {"a": {"b": "包含}大括号"}, "c": [1, 2]}


def test_extract_json_garbage():
    assert extract_json("没有 JSON") is None


# ---------------------------------------------------------------- sanitize
def _ok_obj():
    return {
        "场景大类": "室内",
        "场景小类": ["咖啡厅、花店"],
        "主体大类": "单人",
        "主体小类": "中青年女性",
        "培体": ["杯子", "桌子"],
        "场景特征": "宽敞半开放或封闭空间",
        "运镜top5": [
            {"name": "环绕运镜", "reason": "主体居中"},
            {"name": "前推运镜", "reason": "突出层次"},
            {"name": "主角跟随", "reason": "可跟拍"},
            {"name": "横移运镜", "reason": "展示环境"},
            {"name": "静止", "reason": "构图稳定"},
        ],
    }


def test_sanitize_happy_path():
    labels, top5, issues = sanitize(_ok_obj())
    assert labels["场景类别"] == "室内/咖啡厅、花店"
    assert labels["被拍摄主体"] == "单人/中青年女性"
    assert labels["培体"] == "杯子_桌子"
    assert labels["场景特征"] == "宽敞半开放或封闭空间"
    assert [t["name"] for t in top5] == ["环绕运镜", "前推运镜", "主角跟随", "横移运镜", "静止"]
    assert top5[0]["atomic"] == "orbit"
    assert not need_retry(labels, top5)


def test_sanitize_peiti_out_of_vocab_removed():
    obj = _ok_obj()
    obj["培体"] = ["杯子", "不存在的东西", "桌子"]
    labels, _, issues = sanitize(obj)
    assert labels["培体"] == "杯子_桌子"
    assert any("培体词表外剔除" in i for i in issues)


def test_sanitize_clamps_out_of_scope_minor():
    obj = _ok_obj()
    obj["场景小类"] = ["树林"]  # 室外的值配了室内大类 -> 剔除后小类为空
    labels, top5, issues = sanitize(obj)
    assert labels["场景类别"] == "室内"
    assert any("越界剔除" in i for i in issues)


def test_sanitize_infers_major_from_minor():
    obj = _ok_obj()
    obj["场景大类"] = "错误值"
    labels, _, _ = sanitize(obj)
    assert labels["场景类别"] == "室内/咖啡厅、花店"


def test_sanitize_subject_minor_mismatch():
    obj = _ok_obj()
    obj["主体小类"] = "宠物"  # 物体的小类配了单人大类
    labels, _, issues = sanitize(obj)
    assert labels["被拍摄主体"] == "单人"
    assert any("不符" in i for i in issues)


def test_sanitize_movement_alias_and_dedupe():
    obj = _ok_obj()
    obj["运镜top5"] = [{"name": "跟拍", "reason": "别名"}, {"name": "跟随", "reason": "重复"},
                       {"name": "完全胡说", "reason": "词表外"}]
    _, top5, issues = sanitize(obj)
    assert top5[0]["name"] == "主角跟随"
    assert len([t for t in top5 if t["name"] == "主角跟随"]) == 1
    assert any("词表外剔除" in i for i in issues)


def test_sanitize_peiti_empty_is_legal():
    obj = _ok_obj()
    obj["培体"] = []
    labels, _, _ = sanitize(obj)
    assert labels["培体"] == ""


def test_need_retry_and_finalize():
    labels = {"场景类别": None, "被拍摄主体": None, "培体": "", "场景特征": None}
    assert need_retry(labels, [])
    labels2, top5, fallback = finalize(labels, [], [], True, True)
    assert fallback
    assert labels2["场景类别"] == "室内"
    assert labels2["被拍摄主体"] == "空镜/纯场景"
    assert len(top5) == 5
    assert labels2["场景特征"] in {n for n, _ in T.SCENE_FEATURES}
    # 兜底后所有必填字段非空（need_retry 只用于判断是否需要再次询问模型）
    assert all(v is not None for v in labels2.values())


def test_prompt_contains_peiti_and_movements():
    p = build_prompt()
    assert "培体" in p
    for name in ["雕塑", "围巾", "希区柯克", "百万运镜", "局部特写/物体特写", "交通工具（飞机-火车-车-船）"]:
        assert name in p


# ---------------------------------------------------------------- mock 端到端
class MockClient:
    """返回固定合法 JSON 的假客户端。"""

    model = "mock-vlm"

    def __init__(self, payload=None):
        import json
        self.payload = payload or _ok_obj()
        import json as _json
        self._json = _json

    def vision(self, prompt, image_paths, *, system=None, max_tokens=512, temperature=0.1):
        return "标注结果：" + self._json.dumps(self.payload, ensure_ascii=False)


def test_annotate_one_mock(tmp_path: Path):
    from video_annotator.annotate import annotate_one

    # 造一张 1x1 图
    from PIL import Image

    img = tmp_path / "case.jpg"
    Image.new("RGB", (32, 32), (120, 120, 120)).save(img)

    record = annotate_one(MockClient(), img)
    assert record["status"] == "ok"
    assert record["model"] == "mock-vlm"
    assert record["labels"]["场景类别"] == "室内/咖啡厅、花店"
    assert len(record["movement_top5"]) == 5
    assert record["error"] is None


def test_annotate_one_mock_garbage(tmp_path: Path):
    from video_annotator.annotate import annotate_one

    from PIL import Image

    img = tmp_path / "bad.jpg"
    Image.new("RGB", (32, 32), (10, 10, 10)).save(img)

    class GarbageClient(MockClient):
        def vision(self, prompt, image_paths, **kw):
            return "完全不是 JSON 的回答"

    record = annotate_one(GarbageClient(), img)
    assert record["status"] == "error"
    assert "无法解析" in record["error"]


def test_annotate_one_mock_partial_with_fallback(tmp_path: Path):
    """模型漏掉字段时兜底生效，记录 status=fallback 且字段完整。"""
    from video_annotator.annotate import annotate_one
    from PIL import Image

    img = tmp_path / "partial.jpg"
    Image.new("RGB", (32, 32), (200, 200, 200)).save(img)

    payload = {"场景大类": "室外自然", "场景小类": ["树林"], "主体大类": "空镜",
               "主体小类": "纯场景", "运镜top5": [{"name": "横摇", "reason": "全景"}]}
    record = annotate_one(MockClient(payload), img)
    assert record["status"] == "fallback"
    assert record["labels"]["场景类别"] == "室外自然/树林"
    assert record["labels"]["培体"] == ""
    assert record["labels"]["场景特征"]  # 兜底补全
    assert len(record["movement_top5"]) == 5


# ---------------------------------------------------------------- report
def test_report_generation(tmp_path: Path):
    import json

    from video_annotator.report import generate_report, load_records

    jsonl = tmp_path / "annotations.jsonl"
    recs = [
        {"id": "a.jpg", "model": "m", "status": "ok", "latency_s": 2.0,
         "labels": {"场景类别": "室内/咖啡厅、花店", "被拍摄主体": "单人/中青年女性",
                    "培体": "杯子_桌子", "场景特征": "宽敞半开放或封闭空间"},
         "movement_top5": [{"name": "环绕运镜", "atomic": "orbit", "reason": "r"}]},
        {"id": "b.jpg", "model": "m", "status": "fallback", "latency_s": 3.0,
         "labels": {"场景类别": "室外自然/树林", "被拍摄主体": "空镜/纯场景",
                    "培体": "", "场景特征": "全开放空间"},
         "movement_top5": [{"name": "横摇", "atomic": "pan", "reason": "r"}]},
    ]
    jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    assert len(load_records(jsonl)) == 2

    out = generate_report(jsonl, tmp_path)
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["n"] == 2
    assert report["dims"]["场景大类"] == {"室内": 1, "室外自然": 1}
    assert report["dims"]["培体"]["场景培体"]["桌子"] == 1
    assert report["dims"]["培体"]["人物培体"]["杯子"] == 1
    assert report["dims"]["运镜top1"]["环绕运镜"] == 1
    assert (tmp_path / "summary.csv").exists()


def test_retry_budget_and_selected_raw(tmp_path):
    import json
    from video_annotator.annotate import annotate_one
    class SequenceClient:
        model = 'mock'
        def __init__(self, replies): self.replies, self.calls = replies, 0
        def vision(self, *args, **kwargs):
            reply = self.replies[self.calls]
            self.calls += 1
            return reply
    bad = _ok_obj(); bad['主体小类'] = 'invalid'
    client = SequenceClient(['not json', json.dumps(bad)])
    result = annotate_one(client, tmp_path/'x.jpg', keep_raw=True)
    assert client.calls == 2
    assert result['status'] == 'fallback'
    client = SequenceClient([json.dumps(bad), json.dumps(_ok_obj())])
    result = annotate_one(client, tmp_path/'x.jpg', keep_raw=True)
    assert result['status'] == 'ok'
    assert result['raw'] == _ok_obj()


def test_minor_limit_and_short_top5_retry():
    obj = _ok_obj(); obj['场景小类'] = T.scene_minor_options('室内')[:4]
    labels, top5, issues = sanitize(obj)
    assert len(labels['场景类别'].split('/', 1)[1].split('_')) == 3
    assert issues
    assert need_retry(labels, top5[:4])


def test_model_detection_failure_stays_per_record(tmp_path):
    from video_annotator.annotate import annotate_one
    class BrokenClient:
        @property
        def model(self): raise RuntimeError('offline')
        def vision(self, *args, **kwargs): raise RuntimeError('offline')
    assert annotate_one(BrokenClient(), tmp_path/'x.jpg')['status'] == 'error'


def test_report_preserves_slash_inside_minor(tmp_path):
    import json
    from video_annotator.report import generate_report
    p = tmp_path/'annotations.jsonl'
    p.write_text(json.dumps({'id':'a','status':'ok','labels':{'场景类别':'其他/局部特写/物体特写'}})+'\n')
    report = json.loads(generate_report(p, tmp_path).read_text())
    assert report['dims']['场景小类']['局部特写/物体特写'] == 1


def test_corrupt_records_fail_without_silent_loss(tmp_path):
    from video_annotator.report import load_records
    p = tmp_path/'annotations.jsonl'; p.write_text('{broken}\n')
    with pytest.raises(ValueError): load_records(p)


@pytest.mark.parametrize('force', [False, True])
def test_cli_resume_errors_and_partial_force_preserves_records(tmp_path, monkeypatch, force):
    import json, sys
    import run_annotator as cli
    source = tmp_path/'input'; source.mkdir()
    (source/'a.jpg').write_bytes(b'mock')
    out = tmp_path/'out'; out.mkdir()
    p = out/'annotations.jsonl'
    old = [{'id':'a.jpg','status':'error'}, {'id':'b.jpg','status':'ok','labels':{}}]
    p.write_text(''.join(json.dumps(r)+'\n' for r in old))
    monkeypatch.setattr(sys, 'argv', ['run_annotator','--input',str(source),'--out',str(out),'--limit','1']+(['--force'] if force else []))
    monkeypatch.setattr(cli,'LMSClient',lambda **kw: MockClient())
    assert cli.main() == 0
    records = {r['id']:r for r in cli.load_records(p)}
    assert records['a.jpg']['status'] == 'ok'
    assert records['b.jpg'] == old[1]


def test_image_orientation_applied_before_sending(tmp_path):
    import base64, io
    from PIL import Image
    from video_annotator.llm import image_data_url
    p = tmp_path/'rotated.jpg'
    im = Image.new('RGB',(40,20)); exif=im.getexif(); exif[274]=6
    im.save(p, exif=exif)
    result = Image.open(io.BytesIO(base64.b64decode(image_data_url(p).split(',')[1])))
    assert result.size == (20,40)


@pytest.mark.parametrize("major,old,new", [
    ("室外自然", "沙漠、戈壁、土地（增补）", "沙漠、戈壁、土地"),
    ("室内", "酒店、商场、广场（机场车站）", "酒店、商场、广场"),
    ("室内", "机场车站（分离标注）", "机场车站"),
])
@pytest.mark.parametrize("old_subject,new_subject", [("情侣1", "情侣"), ("亲子1", "亲子")])
def test_confirmed_names_through_annotation(tmp_path, major, old, new, old_subject, new_subject):
    from video_annotator.annotate import annotate_one
    obj = _ok_obj()
    obj.update({"场景大类": major, "场景小类": [old], "主体大类": "多人", "主体小类": old_subject})
    record = annotate_one(MockClient(obj), tmp_path / "mock.jpg", keep_raw=True)
    assert record["status"] == "ok"
    assert record["labels"]["场景类别"] == major + "/" + new
    assert record["labels"]["被拍摄主体"] == "多人/" + new_subject
    assert record["raw"] == obj
    obj.update({"场景小类": [new], "主体小类": new_subject})
    labels, _, issues = sanitize(obj)
    assert not issues
    assert labels == record["labels"]


def test_confirmed_prompt_names_and_semantic_parentheses():
    prompt = build_prompt()
    for old in ["（增补）", "（分离标注）", "广场（机场车站）", "情侣1", "亲子1"]:
        assert old not in prompt
    for name in ["酒店、商场、广场", "机场车站", "情侣", "亲子", "小孩（<6）", "少年（6-15）", "体育场馆（运动、含开阔室内场馆）"]:
        assert name in prompt
        assert T.canonical(name) == name
    assert T.canonical("酒店/商场/广场（机场车站）") == "酒店、商场、广场"
