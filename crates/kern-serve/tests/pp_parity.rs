//! The pipeline head's item-size policies against the simulator they
//! were designed in (`ppsim`, `events=`): every admission, answer and
//! decision of a simulated run replayed through the head's own policy
//! core, each decision's cap and whether its item went out the same.
//!
//! The fixture is the first 400 requests of a K3 admit trace made replayable on
//! DeepSeek V4.1 Flash, compressed nine times, under the four-rung ladder
//! and DSV4.1 PP4's cost model. `PP_PARITY` names longer logs to check the
//! same way, `;`-separated `<events>,<costfit>,<policy>[,<backlog s>]`.

use kern_serve::cost::Cost;
use kern_serve::frontier::{apply, goes_out, pack, Seq, Turn};
use kern_serve::policy::{Meter, Policy};

const STAGES: usize = 4;
const AHEAD: usize = 4;
const CONTEXT: i64 = 1 << 20;

/// Replays `events` under `policy`; the decisions matched by the ladder
/// rung each took, or the first that did not match.
fn replay(events: &str, policy: &Policy, cost: &Cost, backlog: f64) -> Result<Vec<usize>, String> {
    let (mut seqs, mut meter, mut flying, mut req) = (Vec::<Seq>::new(), Meter::new(STAGES, AHEAD, backlog), 0, 0);
    let mut decided = Vec::new();
    for (n, line) in events.lines().enumerate() {
        let f: Vec<&str> = line.split(' ').collect();
        let num = |i: usize| f[i].parse::<f64>().map_err(|e| format!("line {}: {e}", n + 1));
        let int = |i: usize| f[i].parse::<i64>().map_err(|e| format!("line {}: {e}", n + 1));
        match f[0] {
            "a" => {
                let (t, hit, prompt) = (num(1)?, int(2)?, int(3)?);
                seqs.push(Seq::new(req, hit, prompt, 0));
                meter.admit(t, hit, prompt, cost);
                req += 1;
            }
            "r" => {
                flying -= 1;
                meter.forget(num(1)?);
            }
            "d" => {
                let (t, want, sent) = (num(1)?, int(2)?, int(3)? == 1);
                meter.forget(t);
                let d = meter.decision(t, &seqs, flying, cost, CONTEXT);
                let (cap, rung) = (policy.cap(&d), policy.rung(d.load).unwrap_or(0));
                let turn = Turn::of(&seqs, STAGES);
                let (plan, rows) = pack(&seqs, &turn, cap, CONTEXT);
                let goes = goes_out(rows, plan.len(), cap, flying, STAGES, AHEAD);
                if (cap, goes) != (want, sent) {
                    return Err(format!(
                        "line {}, decision {} at {t}: cap {cap} goes {goes}, the simulator {want} {sent}",
                        n + 1,
                        decided.iter().sum::<usize>()
                    ));
                }
                if goes {
                    let (parts, _) = apply(&mut seqs, &plan, &turn.left);
                    meter.sent(t, &parts, cost);
                    flying += 1;
                }
                decided.resize(decided.len().max(rung + 1), 0);
                decided[rung] += 1;
            }
            x => return Err(format!("line {}: unknown event `{x}`", n + 1)),
        }
    }
    Ok(decided)
}

#[test]
fn the_ladder_decides_as_the_simulator_does() {
    let cost = Cost::parse(include_str!("fixtures/pp/dsv41-pp4-cost.json")).unwrap();
    let policy = Policy::parse("ladder").unwrap();
    let rungs = replay(include_str!("fixtures/pp/ladder-load9.events"), &policy, &cost, 0.0).unwrap();
    assert_eq!((rungs.iter().sum::<usize>(), rungs.iter().filter(|&&n| n > 0).count()), (1807, 4), "{rungs:?}");
}

#[test]
fn the_ladder_reading_its_backlog_decides_as_the_simulator_does() {
    let cost = Cost::parse(include_str!("fixtures/pp/dsv41-pp4-cost.json")).unwrap();
    let policy = Policy::parse("ladder").unwrap();
    let rungs = replay(include_str!("fixtures/pp/ladder-backlog2-load9.events"), &policy, &cost, 2.0).unwrap();
    assert_eq!(rungs, [111, 15, 59, 1255], "{rungs:?}");
}

#[test]
fn longer_logs_decide_as_the_simulator_does() {
    let Ok(runs) = std::env::var("PP_PARITY") else { return };
    for run in runs.split(';').filter(|r| !r.is_empty()) {
        let (events, costfit, policy, backlog) = match run.split(',').collect::<Vec<_>>()[..] {
            [e, c, p] => (e, c, p, 0.0),
            [e, c, p, b] => (e, c, p, b.parse().unwrap()),
            _ => panic!("`{run}`"),
        };
        let cost = Cost::parse(&std::fs::read_to_string(costfit).unwrap()).unwrap();
        let policy = Policy::parse(policy).unwrap();
        let decided = replay(&std::fs::read_to_string(events).unwrap(), &policy, &cost, backlog)
            .unwrap_or_else(|e| panic!("{events} under {policy:?}: {e}"));
        eprintln!("{events}: {} decisions match, by rung {decided:?}", decided.iter().sum::<usize>());
    }
}
