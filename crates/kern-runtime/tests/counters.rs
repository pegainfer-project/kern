//! The device's counters as a tracing rank samples them: back to back on
//! the interval asked for, on the clock kernels are stamped in, idle while
//! nothing runs and moving DRAM while a copy does.

use std::time::Duration;

use cudarc::driver::CudaContext;
use kern_runtime::counters::{self, Sample, Sampler};
use kern_runtime::trace;

const EVERY: Duration = Duration::from_micros(100);

fn window(s: &[Sample], f: fn(&Sample) -> f64) -> f64 {
    s.iter().map(f).sum::<f64>() / s.len() as f64
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn samples_tile_the_interval_on_the_trace_clock_and_follow_the_device() {
    trace::start().unwrap();
    let mut sampler = Sampler::start(0, EVERY).unwrap();
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let (src, mut dst) = (stream.alloc_zeros::<u8>(1 << 30).unwrap(), stream.alloc_zeros::<u8>(1 << 30).unwrap());
    stream.synchronize().unwrap();
    sampler.collect().unwrap();

    std::thread::sleep(Duration::from_millis(50));
    let idle = sampler.collect().unwrap();
    (0..100).for_each(|_| stream.memcpy_dtod(&src, &mut dst).unwrap());
    stream.synchronize().unwrap();
    let copying = sampler.collect().unwrap();
    let now = trace::now().unwrap();

    let all = [idle.as_slice(), copying.as_slice()].concat();
    let back_to_back = all.windows(2).all(|w| w[1].start == w[0].end);
    let widths: Vec<u64> = all.iter().map(|s| s.end - s.start).collect();
    let on_interval = widths.iter().all(|&w| w.abs_diff(EVERY.as_nanos() as u64) <= 5_000);
    let last = all.last().unwrap().end;
    assert_eq!(
        (back_to_back, on_interval, idle.len() >= 400, last <= now && now - last < 300_000_000),
        (true, true, true, true)
    );

    let busiest = copying.iter().map(|s| s.dram_read.min(s.dram_write)).fold(0.0, f64::max);
    let (idle_active, clock) = (window(&idle, |s| s.sm_active), window(&idle, |s| s.sm_clock));
    assert_eq!((idle_active < 1.0, clock > 1e8, busiest > 1e11), (true, true, true), "{idle_active} {clock} {busiest}");
    assert_eq!(sampler.overflows(), 0);
}

#[test]
fn an_interval_under_the_floor_is_refused_before_the_device_is_touched() {
    let refused = Sampler::start(0, counters::FLOOR - Duration::from_micros(1)).err().map(|e| e.to_string());
    assert!(refused.as_deref().is_some_and(|e| e.contains("no faster than")), "{refused:?}");
}
