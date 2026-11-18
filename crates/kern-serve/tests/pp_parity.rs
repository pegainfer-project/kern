//! The pipeline head's item-size policies against the simulator they
//! were designed in (`ppsim`, `view=ends`, `events=`): every admission,
//! answer and decision of a simulated run replayed through the head's own
//! policy core, each decision's cap and whether its item went out the
//! same. The head is told what the simulated one hears, no sooner: when
//! stage 0 started and finished each item, and when the last stage
//! finished it, once its answer has had time to come back.
//!
//! The fixtures are the first 400 requests of a K3 admit trace made
//! replayable on DeepSeek V4.1 Flash, compressed nine times, under
//! DSV4.1 PP4's cost model (under SGLang's rule, the first 3000 events of
//! them). `PP_PARITY` names longer logs to check the same way,
//! `;`-separated `<events>,<costfit>,<policy>[,<backlog s>]`.

use kern_serve::cost::Cost;
use kern_serve::frontier::{apply, goes_out, pack, Seq, Turn, REPLY};
use kern_serve::policy::{Meter, Policy};

const STAGES: usize = 4;
const AHEAD: usize = 4;
const CONTEXT: i64 = 1 << 20;

/// Replays `events` under `policy`; the decisions matched by the ladder
/// rung each took (all under rung 0 without a ladder), or the first that
/// did not match.
fn replay(events: &str, policy: &Policy, cost: &Cost, backlog: f64) -> Result<Vec<usize>, String> {
    let (mut seqs, mut meter, mut flying, mut req) = (Vec::<Seq>::new(), Meter::new(STAGES, AHEAD, backlog), 0, 0);
    // Each sent item's stage-0 start, stage-0 finish and exit, and how many of each the head has heard.
    let (mut ran, mut heard) = (Vec::<[f64; 3]>::new(), [0usize; 3]);
    let mut decided = Vec::new();
    for (n, line) in events.lines().enumerate() {
        let f: Vec<&str> = line.split(' ').collect();
        let num = |i: usize| f[i].parse::<f64>().map_err(|e| format!("line {}: {e}", n + 1));
        let int = |i: usize| f[i].parse::<i64>().map_err(|e| format!("line {}: {e}", n + 1));
        match f[0] {
            "a" => {
                let (t, hit, prompt) = (num(1)?, int(2)?, int(3)?);
                seqs.push(Seq::new(req, hit, prompt, t));
                meter.admit(t, hit, prompt, cost);
                req += 1;
            }
            "r" => {
                flying -= 1;
                meter.forget(num(1)?);
            }
            "s" => ran.push([num(1)?, num(2)?, num(3)?]),
            "d" => {
                let (t, want, sent) = (num(1)?, int(2)?, int(3)? == 1);
                for (what, upto) in [t, t, t - REPLY].into_iter().enumerate() {
                    while ran.get(heard[what]).is_some_and(|x| x[what] <= upto) {
                        let (item, at) = (heard[what] as u64, ran[heard[what]][what]);
                        match what {
                            0 => meter.began(item, at),
                            1 => meter.ended(item, at),
                            _ => meter.answered(item, at),
                        }
                        heard[what] += 1;
                    }
                }
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
                    meter.sent(ran.len() as u64, t, &parts, cost);
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

fn fixture(events: &str, policy: &str, backlog: f64) -> Vec<usize> {
    let cost = Cost::parse(include_str!("fixtures/pp/dsv41-pp4-cost.json")).unwrap();
    replay(events, &Policy::parse(policy).unwrap(), &cost, backlog).unwrap()
}

#[test]
fn headway_decides_as_the_simulator_does() {
    let rungs = fixture(include_str!("fixtures/pp/headway-load9.events"), "headway10240/12288", 6.0);
    assert_eq!(rungs, [404, 107, 357, 821], "{rungs:?}");
}

#[test]
fn sglang_dynamic_chunking_decides_as_the_simulator_does() {
    let n = fixture(include_str!("fixtures/pp/dck-load9.events"), "dck8192/0.75", 0.0);
    assert_eq!(n, [979], "{n:?}");
}

#[test]
fn hwang_and_ahn_decide_as_the_simulator_does() {
    let n = fixture(include_str!("fixtures/pp/hwang-load9.events"), "hwang1500/0", 0.0);
    assert_eq!(n, [1284], "{n:?}");
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
