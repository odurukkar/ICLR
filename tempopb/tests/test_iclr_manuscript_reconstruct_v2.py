"""Read-only real-bundle checks plus isolated failure injections; no experiments."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

import iclr_manuscript_reconstruct_v2 as m
from iclr_manuscript_aux_v2 import format_aux_macros


@pytest.fixture(scope="module")
def captured():
    return m.capture_inputs()


def auxiliary(c, consumers=None, contextual=None, rows=None):
    return format_aux_macros(consumers if consumers is not None else c['consumers']['consumers'],
                             rows if rows is not None else c['per_series'],c['split'],c['optimizer'],
                             contextual if contextual is not None else c['contextual'])


def test_complete_inventory_direct_mapping_and_known_deltas(captured):
    p=m.reconstruct(captured)
    assert p['macro_count']==596 and p['direct_map_exact']==69
    assert p['counts']=={'reconstructed_primary_v2':362,'unaffected_non_primary':213,
                         'archived_synthetic_carry_forward':17,'unavailable_v2_runtime':4}
    changed={n:(r['previous_value'],r['value']) for n,r in p['records'].items() if r['changed']}
    assert changed=={'PartAgeRESCSD':('0.1127','0.1106'),'PartSexRESCSD':('0.0585','0.0558'),
                     'AgeLookupVsSeniorCI':('[-0.0381, -0.0079]','[-0.0382, -0.0078]'),
                     'HistoryFreeVsAgeLookupCI':('[-0.0160, +0.0237]','[-0.0168, +0.0235]'),
                     **{n:(captured['working'][n],None) for n in m.RUNTIME_NAMES}}
    for n,v in captured['consumers']['manuscript_macros'].items(): assert p['records'][n]['value']==v
    assert p['records']['ContextualVsAgeLookupDiff']['value']=='+0.0112'
    assert p['records']['ContextualVsAgeLookupP']['value']=='= 0.311'
    assert p['records']['ContextualVsAgeLookupRecord']['value']=='5/1/12'
    assert p['records']['PartAgeMESCSD']['value']=='0.1127'
    assert p['records']['PartSexMESCSD']['value']=='0.0585'
    m.recheck_inputs(captured)


@pytest.mark.parametrize('key,bad',[('soft_target',.85),('endowment_source_id','wrong'),
                                    ('endowment_csd',float('nan')),('direct_csd',float('inf')),
                                    ('endowment_wins',True),('n_series',18.5),
                                    ('endowment_wins',None),('endowment_wins',19),
                                    ('n_series',None)])
def test_aux_rejects_wrong_matched_identity_or_numeric(captured,key,bad):
    cs=deepcopy(captured['consumers']['consumers']);cs['summary.matched_target']['values']['matched_rows'][0][key]=bad
    with pytest.raises(ValueError): auxiliary(captured,consumers=cs)


@pytest.mark.parametrize('field', ['scored_editions', 'editions_with_uncovered_funded_project'])
@pytest.mark.parametrize('bad', [None, True, -1, 54.0])
def test_coverage_rejects_invalid_common_counts(captured, field, bad):
    cs = deepcopy(captured['consumers']['consumers'])
    # All-null scored counts used to satisfy the common-edition check and emit
    # the literal string "None". Keep every row equal to exercise that path.
    for row in cs['mechanism.attribution_coverage']['values']['policy_rows']:
        row[field] = bad
    with pytest.raises(ValueError):
        auxiliary(captured, consumers=cs)


COUNT_LOCATIONS = [
    ('control.senior_scalar', ('selected', 'wins')),
    ('control.senior_scalar', ('selected', 'n_series')),
    ('primary.corpus', ('instance_rows', 0, 'year')),
    ('primary.corpus', ('instance_rows', 0, 'n_voters')),
    ('primary.corpus', ('instance_rows', 0, 'n_projects')),
    ('primary.corpus', ('granularity_rows', 0, 'scored_elections')),
    ('baseline.primary', ('series_count',)),
    ('outcome.per_series', ('learned_rows', 0, 'learned_wins')),
    ('failure.outcome', ('worst_cohort_mix', 'WorstCohortMale', 'numerator')),
    ('failure.outcome', ('worst_cohort_mix', 'WorstCohortMale', 'denominator')),
    ('control.static_age_lookup', ('contrast_rows', 0, 'wins')),
    ('control.static_age_lookup', ('contrast_rows', 0, 'ties')),
    ('control.static_age_lookup', ('contrast_rows', 0, 'n')),
]


@pytest.mark.parametrize('cid,path', COUNT_LOCATIONS)
@pytest.mark.parametrize('bad', [None, True, -1, 1.5])
def test_aux_consumed_counts_require_nonnegative_integers(captured, cid, path, bad):
    cs = deepcopy(captured['consumers']['consumers'])
    target = cs[cid]['values']
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad
    with pytest.raises(ValueError):
        auxiliary(captured, consumers=cs)


@pytest.mark.parametrize('field', ['wins', 'ties', 'losses', 'n_series'])
@pytest.mark.parametrize('bad', [None, True, -1, 1.5])
def test_contextual_consumed_counts_require_nonnegative_integers(captured, field, bad):
    contextual = deepcopy(captured['contextual'])
    contextual['statistics'][field] = bad
    with pytest.raises(ValueError):
        auxiliary(captured, contextual=contextual)


@pytest.mark.parametrize('cid,path,denominator', [
    ('control.senior_scalar', ('selected', 'wins'), 'n_series'),
    ('mechanism.attribution_coverage', ('policy_rows', 0, 'editions_with_uncovered_funded_project'), 'scored_editions'),
    ('failure.outcome', ('worst_cohort_mix', 'WorstCohortMale', 'numerator'), 'denominator'),
    ('control.static_age_lookup', ('contrast_rows', 0, 'wins'), 'n'),
])
def test_aux_count_numerators_cannot_exceed_totals(captured, cid, path, denominator):
    cs = deepcopy(captured['consumers']['consumers'])
    target = cs[cid]['values']
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = target[denominator] + 1
    with pytest.raises(ValueError, match='exceeds'):
        auxiliary(captured, consumers=cs)


def test_age_wins_and_ties_together_cannot_exceed_total(captured):
    cs = deepcopy(captured['consumers']['consumers'])
    row = cs['control.static_age_lookup']['values']['contrast_rows'][0]
    row.update(wins=row['n'], ties=1)
    with pytest.raises(ValueError, match='exceeds'):
        auxiliary(captured, consumers=cs)


def test_contextual_wins_cannot_exceed_total(captured):
    contextual = deepcopy(captured['contextual'])
    contextual['statistics']['wins'] = contextual['statistics']['n_series'] + 1
    with pytest.raises(ValueError, match='exceeds'):
        auxiliary(captured, contextual=contextual)


def test_nullable_unconsumed_fields_remain_allowed(captured):
    cs = deepcopy(captured['consumers']['consumers'])
    cs['primary.corpus']['values']['optional_diagnostic'] = {
        'n_groups': None, 'paired_cohens_dz': None,
    }
    assert auxiliary(captured, consumers=cs) == auxiliary(captured)


def test_age_labels_cannot_exchange_fit_pairs(captured):
    cs=deepcopy(captured['consumers']['consumers']);rows=cs['control.static_age_lookup']['values']['contrast_rows']
    rows[0]['contrast'],rows[1]['contrast']=rows[1]['contrast'],rows[0]['contrast']
    with pytest.raises(ValueError,match='fit identities'): auxiliary(captured,consumers=cs)


def test_wrong_consumer_schema_and_baseline_count(captured):
    for mutate in ('schema','series_count'):
        cs=deepcopy(captured['consumers']['consumers']);cs['baseline.primary']['values'][mutate]='bad' if mutate=='schema' else 999
        with pytest.raises(ValueError): auxiliary(captured,consumers=cs)


def test_pair_requires_all_unique_series(captured):
    rows=[r for r in captured['per_series'] if not (r['fit_id']=='static-senior-grid' and r['view']=='test' and r['series'].endswith('/Bemowo'))]
    with pytest.raises(ValueError,match='paired series'): auxiliary(captured,rows=rows)


def test_contextual_policy_cannot_be_history_free(captured):
    c=deepcopy(captured['contextual']);c['contextual_fit_id']='history_free/temporal_2022/seed-42'
    with pytest.raises(ValueError,match='wrong contextual'): auxiliary(captured,contextual=c)


def test_aux_integer_losses_and_effect_size_provenance(captured):
    r=auxiliary(captured)
    assert r['PerSeriesWins']['value']==captured['working']['PerSeriesWins']
    assert r['AgeLookupVsSeniorDz']['value']==captured['working']['AgeLookupVsSeniorDz']
    assert r['HistoryFreeVsAgeLookupDz']['value']==captured['working']['HistoryFreeVsAgeLookupDz']
    assert all(s['artifact']=='per_series' for s in r['AgeLookupVsSeniorDz']['sources'])


def test_coverage_cannot_silently_drop_or_add_macro(captured,monkeypatch):
    original=m.format_core_macros
    def missing(cs):
        out=original(cs);out.pop('EndowCSD');return out
    monkeypatch.setattr(m,'format_core_macros',missing)
    with pytest.raises(ValueError,match='macro coverage mismatch'): m.reconstruct(captured)


def test_direct_map_disagreement_fails(captured,monkeypatch):
    original=m.format_core_macros
    def changed(cs):
        out=original(cs);out['EndowCSD']['value']='0.9999';return out
    monkeypatch.setattr(m,'format_core_macros',changed)
    with pytest.raises(ValueError,match='direct-map disagreement'): m.reconstruct(captured)


def test_dangling_field_provenance_fails(captured):
    records={'ValidName':{'value':'1','transformation':'test','sources':[
        {'consumer':'summary.matched_target','table':'matched_rows','coordinates':{},'fields':['not_a_field']}]}}
    with pytest.raises(ValueError,match='dangling consumer'): m.validate_sources(records,captured)


def test_candidate_omits_unavailable_timing_definitions(captured):
    output=m.render_outputs(m.reconstruct(captured));tex=output['numbers_candidate.tex'].decode()
    assert len(m.audit.parse_numbers_tex(output['numbers_candidate.tex']))==592
    for name in m.RUNTIME_NAMES:
        assert '\\newcommand{\\'+name+'}' not in tex
        assert 'UNAVAILABLE corrected-v2 runtime: '+name in tex
    decoded=json.loads(output['macro_records.json'])
    assert decoded['records']['DegCSD']['classification']=='archived_synthetic_carry_forward'


def test_recheck_detects_input_mutation(tmp_path):
    path=tmp_path/'input';path.write_bytes(b'old')
    c={'snapshots':{path:m._sha(b'old')}};path.write_bytes(b'new')
    with pytest.raises(ValueError,match='changed during'):m.recheck_inputs(c)


def test_writer_never_overwrites_existing_candidate(captured,tmp_path):
    root=tmp_path/'repo';base=root/m.audit.V2_RESULT_RELATIVE;base.mkdir(parents=True)
    c={**captured,'root':root};p=m.reconstruct(c)
    destination=m.write_outputs(c,p)
    before={f.name:f.read_bytes() for f in destination.iterdir()}
    with pytest.raises(FileExistsError):m.write_outputs(c,p)
    assert before=={f.name:f.read_bytes() for f in destination.iterdir()}
    manifest=json.loads(before['manifest.json'])
    assert all(m._sha(before[n])==h for n,h in manifest['output_sha256'].items())


@pytest.mark.parametrize('target',['source','working','contextual'])
def test_capture_rejects_changed_pinned_source(monkeypatch,target):
    real=m._read
    paths={'source':m.ROOT.parent/m.audit.PROTECTED_SOURCE_RELATIVE,
           'working':m.ROOT.parent/'iclr_paper/tex/numbers.tex',
           'contextual':m.ROOT/'analysis-output/contextual-age-lookup-20260920/summary.json'}
    def injected(path: Path):
        data=real(path)
        return data+b' ' if path==paths[target] else data
    monkeypatch.setattr(m,'_read',injected)
    with pytest.raises(ValueError,match='digest differs'):m.capture_inputs()


@pytest.fixture
def raw_metadata():
    root = m.ROOT / m.audit.V2_RESULT_RELATIVE
    compat_bytes = m._read(root / 'v1_v2_delta_compat_audit.json')
    assert m._sha(compat_bytes) == m.COMPAT_SHA
    bindings = json.loads(compat_bytes)['compatibility_amendment']['sealed_file_sha256']
    lock_bytes = m._read(root / 'protocol_lock.json')
    manifest_bytes = m._read(root / 'corpus_manifest.json')
    for name, content in [('protocol_lock.json', lock_bytes), ('corpus_manifest.json', manifest_bytes)]:
        assert m._sha(content) == bindings['repo/' + (m.audit.V2_RESULT_RELATIVE / name).as_posix()]
    return bindings, json.loads(lock_bytes), json.loads(manifest_bytes), m._sha(manifest_bytes)


def test_raw_allowlist_requires_exact_three_way_authenticated_inventory(raw_metadata):
    allowed = m._authenticated_raw_bindings(*raw_metadata)
    assert len(allowed) == 132
    assert all(name.startswith('repo/data/pb/') and name.endswith('.pb') for name in allowed)


@pytest.mark.parametrize('mutation', [
    'missing_sealed', 'extra_sealed', 'missing_lock', 'duplicate_manifest',
    'missing_manifest', 'extra_manifest', 'sealed_digest', 'lock_digest',
    'manifest_digest', 'manifest_lock_path', 'manifest_lock_digest',
    'raw_lock_label', 'source_dir', 'destination', 'source_relative',
])
def test_raw_allowlist_rejects_metadata_disagreement(raw_metadata, mutation):
    bindings, lock, manifest, digest = raw_metadata
    row = manifest['files'][0]
    sealed_name = 'repo/data/pb/' + row['name']
    locked_name = 'data/' + row['name']
    if mutation == 'missing_sealed': del bindings[sealed_name]
    elif mutation == 'extra_sealed': bindings['repo/data/pb/unlocked.pb'] = row['sha256']
    elif mutation == 'missing_lock': del lock['tracked_files'][locked_name]
    elif mutation == 'duplicate_manifest': manifest['files'][1] = deepcopy(row)
    elif mutation == 'missing_manifest': manifest['files'].pop()
    elif mutation == 'extra_manifest': manifest['files'].append(deepcopy(row))
    elif mutation == 'sealed_digest': bindings[sealed_name] = '0' * 64
    elif mutation == 'lock_digest': lock['tracked_files'][locked_name]['sha256'] = '0' * 64
    elif mutation == 'manifest_digest': row['sha256'] = '0' * 64
    elif mutation == 'manifest_lock_path': lock['tracked_files']['artifact/corpus_manifest']['path'] = '/private/manifest.json'
    elif mutation == 'manifest_lock_digest': lock['tracked_files']['artifact/corpus_manifest']['sha256'] = '0' * 64
    elif mutation == 'raw_lock_label': lock['tracked_files']['data/wrong.pb'] = lock['tracked_files'].pop(locked_name)
    elif mutation == 'source_dir': manifest['source_dir'] = '/private/data/pb'
    elif mutation == 'destination': manifest['destination'] = 'data/elsewhere'
    elif mutation == 'source_relative': row['source_relative'] = '../' + row['name']
    with pytest.raises(ValueError):
        m._authenticated_raw_bindings(bindings, lock, manifest, digest)


@pytest.mark.parametrize('field', ['n_files', 'n_elections'])
@pytest.mark.parametrize('value', [None, True, 132.0, 131, 133])
def test_raw_allowlist_rejects_wrong_counts(raw_metadata, field, value):
    raw_metadata[2][field] = value
    with pytest.raises(ValueError, match='inventory'):
        m._authenticated_raw_bindings(*raw_metadata)


@pytest.mark.parametrize('name', [
    '../outside.pb', '/private/outside.pb', 'nested/file.pb', r'..\outside.pb',
    'wrong.csv', 'nested/../flat.pb', './flat.pb', 'nested//file.pb', 'bad\0.pb',
])
def test_raw_allowlist_rejects_adversarial_paths_even_if_all_metadata_agree(raw_metadata, name):
    bindings, lock, manifest, digest = raw_metadata
    row = manifest['files'][0]
    old_name = row['name']
    row['name'] = row['source_relative'] = name
    bindings['repo/data/pb/' + name] = bindings.pop('repo/data/pb/' + old_name)
    locked = lock['tracked_files'].pop('data/' + old_name)
    locked['path'] = 'data/pb/' + name
    lock['tracked_files']['data/' + name] = locked
    with pytest.raises(ValueError, match='path'):
        m._authenticated_raw_bindings(bindings, lock, manifest, digest)


def test_default_missing_raw_rejected_and_only_opted_raw_may_be_omitted(tmp_path):
    name = 'repo/data/pb/ballot.pb'
    digest = m._sha(b'ballots')
    with pytest.raises(RuntimeError):
        m._capture_sealed_binding(tmp_path, name, digest, {}, {})
    snapshots = {}
    assert m._capture_sealed_binding(tmp_path, name, digest, snapshots, {name: digest}) == {
        'path': name, 'sha256': digest,
    }
    assert snapshots == {}
    with pytest.raises(RuntimeError):
        m._capture_sealed_binding(tmp_path, 'repo/src/missing.py', digest, {}, {name: digest})
    with pytest.raises(ValueError, match='digest differs'):
        m._capture_sealed_binding(tmp_path, name, '0' * 64, {}, {name: digest})


@pytest.mark.parametrize('kind', ['directory', 'symlink', 'broken_symlink', 'parent_symlink', 'parent_file'])
def test_optional_raw_inputs_reject_nonregular_paths(tmp_path, kind):
    name = 'repo/data/pb/ballot.pb'
    digest = m._sha(b'ballots')
    path = tmp_path / 'data/pb/ballot.pb'
    if kind == 'parent_file':
        (tmp_path / 'data').write_bytes(b'not a directory')
    elif kind == 'parent_symlink':
        (tmp_path / 'data').symlink_to(tmp_path / 'absent', target_is_directory=True)
    else:
        path.parent.mkdir(parents=True)
        if kind == 'directory': path.mkdir()
        elif kind == 'broken_symlink': path.symlink_to(tmp_path / 'absent')
        else:
            target = tmp_path / 'target'
            target.write_bytes(b'ballots')
            path.symlink_to(target)
    with pytest.raises(ValueError, match='symlink|file type'):
        m._capture_sealed_binding(tmp_path, name, digest, {}, {name: digest})


def test_optional_present_raw_bytes_are_still_hashed(tmp_path):
    name = 'repo/data/pb/ballot.pb'
    path = tmp_path / 'data/pb/ballot.pb'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'ballots')
    digest = m._sha(b'ballots')
    snapshots = {}
    assert m._capture_sealed_binding(tmp_path, name, digest, snapshots, {name: digest}) is None
    assert snapshots == {path: digest}
    path.write_bytes(b'different')
    with pytest.raises(ValueError, match='digest differs'):
        m._capture_sealed_binding(tmp_path, name, digest, {}, {name: digest})


def test_default_full_capture_still_rejects_missing_raw(monkeypatch, raw_metadata):
    name = raw_metadata[2]['files'][0]['name']
    path = m.ROOT / 'data/pb' / name
    real_read = m._read
    def missing(target):
        if target == path:
            raise RuntimeError('injected raw file is missing')
        return real_read(target)
    monkeypatch.setattr(m, '_read', missing)
    with pytest.raises(RuntimeError, match='injected raw file is missing'):
        m.capture_inputs()


def test_saved_mode_with_raw_present_preserves_exact_saved_numerical_payload(captured):
    saved = m.capture_inputs(allow_missing_raw_data=True)
    assert saved['omitted_raw_data'] == []
    assert saved['allow_missing_raw_data'] is True
    payload = m.reconstruct(saved)
    assert payload == m.reconstruct(captured)
    existing = m._read(m.ROOT / m.audit.V2_RESULT_RELATIVE / 'manuscript_reconstruction/macro_records.json')
    assert m.render_outputs(payload)['macro_records.json'] == existing
    m.recheck_inputs(saved)


def test_saved_mode_all_raw_omissions_reported_without_changing_payload(captured, monkeypatch, raw_metadata):
    allowed = m._authenticated_raw_bindings(*raw_metadata)
    monkeypatch.setattr(m, '_raw_path_missing', lambda root, name: name in allowed)
    real_read = m._read
    def no_raw_reads(path):
        if path.parent == m.ROOT / 'data/pb':
            raise AssertionError('omitted raw file was opened')
        return real_read(path)
    monkeypatch.setattr(m, '_read', no_raw_reads)
    saved = m.capture_inputs(allow_missing_raw_data=True)
    assert saved['omitted_raw_data'] == [{'path': name, 'sha256': allowed[name]} for name in sorted(allowed)]
    assert all(path.parent != m.ROOT / 'data/pb' for path in saved['snapshots'])
    assert m.reconstruct(saved) == m.reconstruct(captured)
    m.recheck_inputs(saved)


def test_saved_mode_rejects_missing_code(monkeypatch):
    real_read = m._read
    def missing(path):
        if path == m.ROOT / 'src/iclr_cmaes.py':
            raise RuntimeError('required code is missing')
        return real_read(path)
    monkeypatch.setattr(m, '_read', missing)
    with pytest.raises(RuntimeError, match='required code is missing'):
        m.capture_inputs(allow_missing_raw_data=True)


def test_recheck_rejects_newly_appeared_omitted_raw_file(tmp_path):
    name = 'repo/data/pb/ballot.pb'
    c = {'root': tmp_path, 'snapshots': {}, 'omitted_raw_data': [{'path': name, 'sha256': m._sha(b'ballots')}]}
    m.recheck_inputs(c)
    path = tmp_path / 'data/pb/ballot.pb'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'ballots')
    with pytest.raises(ValueError, match='appeared during reconstruction'):
        m.recheck_inputs(c)


def test_saved_mode_cannot_write_even_with_no_omissions(captured):
    with pytest.raises(ValueError, match='read-only'):
        m.write_outputs({**captured, 'allow_missing_raw_data': True}, {})


def test_cli_rejects_saved_mode_with_write_before_capture(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['reconstruct', '--saved-results-only', '--write'])
    monkeypatch.setattr(m, 'capture_inputs', lambda **kwargs: pytest.fail('capture must not run'))
    with pytest.raises(SystemExit) as caught:
        m.main()
    assert caught.value.code == 2


def test_cli_reports_every_omitted_raw_path_and_digest(monkeypatch, capsys):
    omitted = [{'path': 'repo/data/pb/a.pb', 'sha256': 'a' * 64},
               {'path': 'repo/data/pb/b.pb', 'sha256': 'b' * 64}]
    c = {'omitted_raw_data': omitted}
    def capture(**kwargs):
        assert kwargs == {'allow_missing_raw_data': True}
        return c
    monkeypatch.setattr(sys, 'argv', ['reconstruct', '--saved-results-only'])
    monkeypatch.setattr(m, 'capture_inputs', capture)
    monkeypatch.setattr(m, 'reconstruct', lambda data: {'macro_count': 596, 'counts': {}, 'direct_map_exact': 69})
    monkeypatch.setattr(m, 'recheck_inputs', lambda data: None)
    monkeypatch.setattr(m, 'write_outputs', lambda *args: pytest.fail('saved mode must not write'))
    m.main()
    assert json.loads(capsys.readouterr().out)['omitted_raw_data'] == omitted
