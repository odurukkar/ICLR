"""Pure auxiliary manuscript formatting from authenticated saved v2 evidence."""
from __future__ import annotations

import math
import statistics

SCHEMAS = {
    "summary.matched_target":"matched-target-contrast-v2",
    "mechanism.attribution_coverage":"attribution-coverage-v2",
    "control.senior_scalar":"senior-scalar-control-v2",
    "control.history_free":"history-free-control-v2",
    "primary.corpus":"primary-corpus-inventory-v2",
    "baseline.primary":"primary-baseline-table-v2",
    "outcome.per_series":"outcome-series-tables-v2",
    "failure.outcome":"outcome-failure-diagnostics-v2",
    "frontier.outcome":"outcome-frontier-v2",
    "summary.endowment_seed_stability":"endowment-seed-stability-v2",
    "control.static_age_lookup":"static-age-lookup-control-v2",
    "outcome.significance":"outcome-significance-v2",
    "summary.endowment_significance":"endowment-significance-suite-v2",
}
NUMERIC_FIELDS = {"scored_editions","uncovered_budget_share","editions_with_uncovered_funded_project",
    "alpha","difference","diff","ci_low","ci_high","ci_lo","ci_hi","p_value","p_two_sided",
    "p_two_sided_exact_sign_flip","wins","ties","losses","n","n_series","seed","floor","soft_target",
    "endowment_csd","direct_csd","endowment_exclusion","direct_exclusion","endowment_wins",
    "year","n_voters","n_projects","median_projects","scored_elections","series_count",
    "learned_wins","numerator","denominator","welfare","exclusion","mean_csd","mean_exclusion",
    "worst_csd","paired_cohens_dz","paired_difference_sd","n_groups"}
COUNT_FIELDS = {"scored_editions","editions_with_uncovered_funded_project","wins","ties","losses",
    "n","n_series","seed","endowment_wins","year","n_voters","n_projects","scored_elections",
    "series_count","learned_wins","numerator","denominator","n_groups"}


def _finite_tree(value: object, key: str = "") -> None:
    if isinstance(value, dict):
        for k,v in value.items(): _finite_tree(v,k)
    elif isinstance(value,list):
        for v in value: _finite_tree(v,key)
    elif isinstance(value,float) and not math.isfinite(value):
        raise ValueError(f"nonfinite input: {key}")
    elif isinstance(value,bool) and (key in NUMERIC_FIELDS or key.startswith(("test_","train_"))):
        raise ValueError(f"boolean numeric input: {key}")
    elif key in COUNT_FIELDS and value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"invalid integer count: {key}")


def _p(value: float) -> str:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("invalid p-value")
    return "< 0.001" if value < .001 else f"= {value:.3f}"


def _one(rows: list[dict], **where: object) -> dict:
    found = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
    if len(found) != 1:
        raise ValueError(f"expected exactly one row: {where}, found {len(found)}")
    return found[0]


def _source(cid: str, table: str, fields: list[str], **where: object) -> dict:
    return {"consumer": cid, "table": table, "coordinates": where, "fields": fields}


def _required_count(value: object, field: str) -> int:
    """Validate a count used in a macro without banning unrelated nullable fields."""
    if type(value) is not int or value < 0:
        raise ValueError(f"invalid required integer count: {field}")
    return value


def _count_group(row: dict, denominator: str, *numerators: str) -> None:
    total = _required_count(row.get(denominator), denominator)
    parts = [_required_count(row.get(field), field) for field in numerators]
    if sum(parts) > total:
        raise ValueError(f"count numerator exceeds {denominator}: {', '.join(numerators)}")


def format_aux_macros(consumers: dict, per_series: list[dict], split: dict,
                      optimizer_source: str, contextual: dict) -> dict[str, dict]:
    """Format saved contrasts; only dz/Holm and descriptive reductions are derived.

    No fitting, allocation replay, bootstrap, or sign-flip test is executed.
    Caller authenticates every input and binds artifact references to hashes.
    """
    result: dict[str, dict] = {}
    for tree in (consumers, per_series, split, contextual): _finite_tree(tree)

    def val(cid: str, table: str) -> object:
        values=consumers[cid]["values"]
        if consumers[cid].get("consumer_id") != cid or values.get("schema") != SCHEMAS[cid]:
            raise ValueError(f"unexpected schema for {cid}")
        return values[table]

    def put(name: str, value: str, sources: list[dict], transform: str) -> None:
        if name in result or not isinstance(value, str):
            raise ValueError(f"duplicate or invalid macro {name}")
        result[name] = {"value": value, "sources": sources, "transformation": transform}

    def consume(name: str, value: str, cid: str, table: str, fields: list[str],
                transform: str, **where: object) -> None:
        put(name, value, [_source(cid, table, fields, **where)], transform)

    cid = "summary.matched_target"
    rows = val(cid, "matched_rows")
    if len(rows) != 1:
        raise ValueError("matched-target row cardinality changed")
    r = rows[0]
    _count_group(r, "n_series", "endowment_wins")
    if (r["soft_target"] != 1.0 or r["n_series"] != 18 or
        r["endowment_source_id"] != "endowment_frontier/temporal_2022/target-1/seed-42" or
        r["direct_source_id"] != "outcome_frontier/temporal_2022/target-1/seed-42"):
        raise ValueError("wrong matched-target policy identities")
    for suffix, field in {"EndowCSD":"endowment_csd", "DirectCSD":"direct_csd",
                          "CILow":"ci_low", "CIHigh":"ci_high",
                          "EndowExcl":"endowment_exclusion",
                          "DirectExcl":"direct_exclusion"}.items():
        consume("MatchedTarget"+suffix, f"{r[field]:.4f}", cid, "matched_rows", [field], "four decimals")
    consume("MatchedTargetDiff", f"{r['difference']:+.4f}", cid, "matched_rows", ["difference"], "signed four decimals")
    consume("MatchedTargetP", _p(r["p_value"]), cid, "matched_rows", ["p_value"], "p threshold .001; otherwise three decimals")
    consume("MatchedTargetWins", f"{r['endowment_wins']}/{r['n_series']}", cid, "matched_rows", ["endowment_wins","n_series"], "wins/series")

    cid = "mechanism.attribution_coverage"
    rows = val(cid, "policy_rows")
    for row in rows:
        _count_group(row, "scored_editions", "editions_with_uncovered_funded_project")
    if len(rows) != 4 or len({r["policy"] for r in rows}) != 4 or len({r["scored_editions"] for r in rows}) != 1:
        raise ValueError("coverage policy cardinality or editions differ")
    for name, value, fields, trans in [
        ("CoverageScoredEditions", str(rows[0]["scored_editions"]), ["scored_editions"], "common scored editions"),
        ("CoverageMaxUncoveredShare", f"{100*max(r['uncovered_budget_share'] for r in rows):.2f}\\%", ["uncovered_budget_share"], "max share times 100, two decimals"),
        ("CoverageAffectedEditions", str(max(r["editions_with_uncovered_funded_project"] for r in rows)), ["editions_with_uncovered_funded_project"], "maximum across policies"),
        ("CoveragePolicyN", str(len(rows)), ["policy"], "number of unique policies")]:
        consume(name, value, cid, "policy_rows", fields, trans)

    cid = "control.senior_scalar"; senior = val(cid, "selected")
    _count_group(senior, "n_series", "wins")
    for suffix, field, digits in [("Alpha","alpha",2),("CSD","test_worst_csd",4),
                                  ("MES","mes_worst_csd",4),("CILow","ci_low",4),("CIHigh","ci_high",4)]:
        consume("Tilt"+suffix, f"{senior[field]:.{digits}f}", cid, "selected", [field], f"{digits} decimals")
    consume("TiltDiff", f"{senior['difference']:+.4f}", cid,"selected",["difference"],"signed four decimals")
    consume("TiltP", _p(senior["p_value"]),cid,"selected",["p_value"],"p formatting")
    consume("TiltWins",f"{senior['wins']}/{senior['n_series']}",cid,"selected",["wins","n_series"],"wins/series")
    cid = "control.history_free"
    r = _one(val(cid,"policy_rows"), policy="static-only (no history)")
    for suffix, field, spec in [("CSD","test_worst_csd",".4f"),("Welfare","test_welfare",",.0f"),("Excl","test_exclusion",".4f")]:
        consume("StaticTilt"+suffix,format(r[field],spec),cid,"policy_rows",[field],spec,policy=r["policy"])

    cid = "primary.corpus"; inst = val(cid,"instance_rows")
    for row in inst:
        for field in ("year", "n_voters", "n_projects"):
            _required_count(row.get(field), field)
    if len(inst) != 132 or len({(r["series"],r["year"]) for r in inst}) != len(inst):
        raise ValueError("primary instance inventory differs")
    for name, value, fields, trans in [
        ("PrimaryElectionN", str(len(inst)), ["file"], "count elections"),
        ("PrimaryBallotN", f"{sum(r['n_voters'] for r in inst):,}",["n_voters"],"sum voters, comma-separated integer"),
        ("PrimaryProjectN",f"{sum(r['n_projects'] for r in inst):,}",["n_projects"],"sum projects, comma-separated integer"),
        ("PrimaryYearLo",str(min(r["year"] for r in inst)),["year"],"minimum year"),
        ("PrimaryYearHi",str(max(r["year"] for r in inst)),["year"],"maximum year")]:
        consume(name,value,cid,"instance_rows",fields,trans)
    gran = _one(val(cid,"granularity_rows"),corpus="warsaw_fitting")
    _required_count(gran.get("scored_elections"), "scored_elections")
    consume("WarsawMedianProjects",f"{gran['median_projects']:.1f}",cid,"granularity_rows",["median_projects"],"one decimal",corpus="warsaw_fitting")
    consume("WarsawScoredElections",str(gran["scored_elections"]),cid,"granularity_rows",["scored_elections"],"integer",corpus="warsaw_fitting")
    if split.get("name") != "temporal_2022" or len(split["test"]) != 18:
        raise ValueError("wrong temporal split")
    split_source = {"artifact":"split", "coordinates":{}, "fields":["train","test"]}
    for which, tag in [("train","Train"),("test","Test")]:
        put("Split"+tag+"Editions",str(sum(len(years) for _,years in split[which])),[split_source],"sum declared split edition counts")
    test_series = [s for s,_ in split["test"]]
    if len(set(test_series)) != len(test_series):
        raise ValueError("duplicate split series")
    cities = sorted({s.split("/")[1] for s in test_series})
    put("EvalSeriesN",str(len(test_series)),[split_source],"number of test series")
    put("EvalCityN",str(len(cities)),[split_source],"number of distinct test-series cities")
    put("EvalCityList",", ".join(cities),[split_source],"sorted city names")
    start = max(min(r["year"] for r in inst if r["series"] == s) for s in test_series)
    put("DynamicsStartYear",str(start),[split_source,_source(cid,"instance_rows",["series","year"])],"maximum of earliest corpus year among evaluated series")
    baseline_rows=val("baseline.primary","series_rows")
    baseline_count = _required_count(val("baseline.primary", "series_count"), "series_count")
    if len(baseline_rows)!=baseline_count or len({r["series"] for r in baseline_rows}) != len(baseline_rows):
        raise ValueError("baseline series count or unique identities differ")
    consume("NumSeriesTotal",str(len(baseline_rows)),"baseline.primary","series_rows",["series"],"count unique baseline series, checked against saved series_count")
    put("OptimizerLines",str(sum(bool(l.strip()) and not l.strip().startswith("#") for l in optimizer_source.splitlines())),
        [{"artifact":"optimizer", "coordinates":{}, "fields":["source_lines"]}],"count nonblank noncomment source lines")
    cid = "outcome.per_series"; per = val(cid,"learned_rows")
    if len(per) != 18 or len({r["series"] for r in per}) != len(per) or any(type(r["learned_wins"]) is not int or r["learned_wins"] not in (0,1) for r in per):
        raise ValueError("invalid per-series win inventory")
    losses = [r for r in per if r["learned_wins"] == 0]
    for name,value in [("PerSeriesN",str(len(per))),("PerSeriesWins",str(len(per)-len(losses))),
                       ("PerSeriesLosses",str(len(losses))),
                       ("LossSeriesList",", ".join(r["series"].split("/")[-1] for r in losses) or "none")]:
        consume(name,value,cid,"learned_rows",["learned_wins","series"],"legacy order; non-wins count as losses")
    for name, counts in val("failure.outcome","worst_cohort_mix").items():
        _count_group(counts, "denominator", "numerator")
        consume(name,f"{counts['numerator']}/{counts['denominator']}","failure.outcome","worst_cohort_mix",[name],"numerator/denominator")

    # Common baseline and primary endowment identities, not the endowment frontier.
    mes = _one(val("frontier.outcome","baseline_rows"),policy="mes",view="test")
    endow = _one(val("summary.endowment_seed_stability","seed_rows"),seed=42)
    cid = "control.static_age_lookup"; age = _one(val(cid,"policy_rows"),policy="age_lookup")
    for name,field,spec in [("AgeLookupCSD","mean_csd",".4f"),("AgeLookupExcl","mean_exclusion",".4f"),("AgeLookupWelfare","welfare",",.0f")]:
        consume(name,format(age[field],spec),cid,"policy_rows",[field],spec,policy="age_lookup")
    asrc = _source(cid,"policy_rows",["welfare","mean_exclusion"],policy="age_lookup")
    msrc = _source("frontier.outcome","baseline_rows",["welfare","exclusion"],policy="mes",view="test")
    esrc = _source("summary.endowment_seed_stability","seed_rows",["test_exclusion"],seed=42)
    if mes["welfare"] <= 0:
        raise ValueError("invalid MES welfare denominator")
    put("AgeLookupWelfareCostPct",f"{100*(1-age['welfare']/mes['welfare']):.1f}\\%",[asrc,msrc],"100*(1-age/MES welfare), one decimal")
    put("AgeLookupExclVsEndowDiff",f"{age['mean_exclusion']-endow['test_exclusion']:+.4f}",[asrc,esrc],"age exclusion minus primary endowment exclusion")
    put("AgeLookupExclVsMESDiff",f"{age['mean_exclusion']-mes['exclusion']:+.4f}",[asrc,msrc],"age exclusion minus MES exclusion")
    weights = val(cid,"fit_record")["selected_weights"]
    if len(weights) != 4:
        raise ValueError("age lookup must have four weights")
    consume("AgeLookupMultipliers",", ".join(f"{math.exp(w):.4f}" for w in weights),cid,"fit_record",["selected_weights"],"elementwise exp, four decimals")
    for key, tag in [("age_lookup_minus_static_60plus","AgeLookupVsSenior"),("history_free_minus_age_lookup","HistoryFreeVsAgeLookup")]:
        r = _one(val(cid,"contrast_rows"),contrast=key)
        pair={"age_lookup_minus_static_60plus":("static_age_lookup/temporal_2022/seed-42","static-senior-grid"),
              "history_free_minus_age_lookup":("history_free/temporal_2022/seed-42","static_age_lookup/temporal_2022/seed-42")}[key]
        if (r["learned_source_id"],r["baseline_source_id"]) != pair:
            raise ValueError("wrong age-contrast fit identities")
        _count_group(r, "n", "wins", "ties")
        if r["n"] != 18 or not 0 <= r["wins"]+r["ties"] <= r["n"]:
            raise ValueError("invalid age-contrast counts")
        values = {"Diff":f"{r['diff']:+.4f}","CI":f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]",
                  "P":_p(r["p_two_sided"]),"Wins":f"{r['wins']}/{r['n']}",
                  "Record":f"{r['wins']}/{r['ties']}/{r['n']-r['wins']-r['ties']}"}
        for suffix,value in values.items():
            consume(tag+suffix,value,cid,"contrast_rows",["diff","ci_lo","ci_hi","p_two_sided","wins","ties","n"],"saved paired contrast; losses=n-wins-ties",contrast=key)
        paired = []
        for fit_id in [r["learned_source_id"],r["baseline_source_id"]]:
            selected = [x for x in per_series if x["fit_id"] == fit_id and x["split"] == "temporal_2022" and x["view"] == "test" and x["scheme"] == "age_sex"]
            mapping = {x["series"]:x["metrics"]["worst_csd"] for x in selected}
            if len(selected) != 18 or len(mapping) != 18 or set(mapping) != set(test_series):
                raise ValueError(f"age contrast lacks exact paired series: {fit_id}")
            paired.append(mapping)
        differences = [paired[0][s]-paired[1][s] for s in sorted(test_series)]
        mean = statistics.mean(differences); sd = statistics.stdev(differences)
        if not math.isclose(mean,r["diff"],abs_tol=1e-12) or sd <= 0:
            raise ValueError("paired contrast identity or variance differs")
        sources = [{"artifact":"per_series", "coordinates":{"fit_id":x,"split":"temporal_2022","view":"test","scheme":"age_sex"},"fields":["series","metrics.worst_csd"]} for x in [r["learned_source_id"],r["baseline_source_id"]]]
        put(tag+"Dz",f"{mean/sd:+.3f}",sources,"paired mean / sample standard deviation (ddof=1); signed three decimals")

    osig = _one(val("outcome.significance","contrast_rows"),baseline="mes",floor=1.0)
    esig = _one(val("summary.endowment_significance","contrast_rows"),contrast="learned-endowment vs mes")
    ps = [osig["p_two_sided"],esig["p_two_sided"],senior["p_value"]]
    adjusted = [0.0]*3; running=0.0
    for rank,i in enumerate(sorted(range(3),key=lambda i:ps[i])):
        _p(ps[i]); running=max(running,(3-rank)*ps[i]);adjusted[i]=min(1.0,running)
    sources=[_source("outcome.significance","contrast_rows",["p_two_sided"],baseline="mes",floor=1.0),
             _source("summary.endowment_significance","contrast_rows",["p_two_sided"],contrast="learned-endowment vs mes"),
             _source("control.senior_scalar","selected",["p_value"])]
    put("WarsawHolmFamilyN","3",sources,"three preexisting focal contrasts")
    for name,p in zip(["SigHeadHolmP","EndowSigMESHolmP","TiltHolmP"],adjusted):
        put(name,_p(p),sources,"Holm step-down adjustment of three saved p-values; no new test")
    r=contextual["statistics"]
    _count_group(r, "n_series", "wins", "ties", "losses")
    if contextual["contextual_fit_id"] != "temporal_endowment/temporal_2022/seed-42" or contextual["age_lookup_fit_id"] != "static_age_lookup/temporal_2022/seed-42" or r["n_series"] != 18:
        raise ValueError("wrong contextual contrast")
    if r["wins"]+r["ties"]+r["losses"] != r["n_series"] or not r["ci_low"] <= r["difference"] <= r["ci_high"]:
        raise ValueError("invalid contextual contrast counts or interval")
    values={"Diff":f"{r['difference']:+.4f}","CI":f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]",
            "P":_p(r["p_two_sided_exact_sign_flip"]),"Record":f"{r['wins']}/{r['ties']}/{r['losses']}",
            "Wins":f"{r['wins']}/{r['n_series']}","Dz":f"{r['paired_cohens_dz']:+.3f}"}
    for suffix,value in values.items():
        put("ContextualVsAgeLookup"+suffix,value,[{"artifact":"contextual", "coordinates":{}, "fields":["statistics"]}],"format saved September 20 corrected-v2 contrast; no recomputation")
    return result
