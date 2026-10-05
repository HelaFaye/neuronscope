import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from scale_sweep_lab import parse_scales, scale_tag, wilson_interval, parse_eval_json, activation_profile_data


def test_scale_grid():
    assert parse_scales(0.2, 0.4, 0.1) == [0.2, 0.3, 0.4]
    assert parse_scales(0.8, 1.2, 0.1) == [0.8, 0.9, 1.0, 1.1, 1.2]


def test_scale_tag():
    assert scale_tag(0.2) == 'supp200'
    assert scale_tag(0.35) == 'supp350'
    assert scale_tag(1.0) == 'base100'
    assert scale_tag(1.2) == 'amp1200'


def test_wilson_interval():
    lo, hi = wilson_interval(80, 100)
    assert 0.70 < lo < 0.77
    assert 0.83 < hi < 0.89


def test_eval_json():
    d = parse_eval_json('{"correct":80,"total":100}')
    assert d['score'] == 0.8
    assert 0 < d['lower_ci'] < 0.8 < d['upper_ci'] < 1


def test_activation_profile(tmp_path):
    p = tmp_path / 'h.json'
    p.write_text(json.dumps({'n_layers': 3, 'n_neurons': 8, 'by_layer': {'0':[1,3], '2':[7]}}))
    d = activation_profile_data(str(p))
    assert d['n_layers'] == 3
    assert d['layers'][0]['selected'] == 2
    assert d['layers'][1]['selected'] == 0
