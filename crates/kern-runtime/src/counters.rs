//! The GPU's own counters, sampled on a fixed interval beside the kernel
//! trace: SM clock, SM and tensor-pipe activity, DRAM reads and writes.
//!
//! [`Sampler::start`] turns on CUPTI's PM sampling for one device: every
//! interval the GPU writes its performance-monitor counters into a
//! hardware buffer by itself, so nothing is inserted between kernels and
//! no kernel is replayed. [`Sampler::collect`] decodes what the buffer
//! holds into [`Sample`]s, each the counters' average over its window,
//! stamped in CUPTI's clock: the one [`crate::trace`] stamps kernels in, so
//! a sample lines up with the kernels that ran inside it.
//!
//! A sample is device-wide; which step or call it belongs to is a join on
//! time where the samples are read. Costs and CUPTI's pitfalls:
//! docs/runtime.md "Counters".

use std::ffi::{c_char, CStr};
use std::time::Duration;

use cudarc::cupti::sys as cupti;

use crate::trace::ok;
use crate::{Error, Result};

/// In [`Sample`]'s field order.
const METRICS: [&CStr; 5] = [
    c"gpc__cycles_elapsed.avg.per_second",
    c"sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
    c"sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed",
    c"dram__bytes_read.sum.per_second",
    c"dram__bytes_write.sum.per_second",
];
/// The shortest interval [`Sampler::start`] takes.
pub const FLOOR: Duration = Duration::from_micros(50);
/// Raw records waiting to be decoded, in host memory: a sample's record is
/// several KB, so seconds of them; past it the oldest are overwritten.
const HARDWARE_BUFFER: usize = 512 << 20;
/// What one collect decodes at most: a few collects' worth. A decode that
/// fills the counter data image hands back no samples on later decodes for
/// a while, and every decode costs more the larger the image (at 100 µs,
/// 22 µs a sample with room for 0.3 s, 36 with 2 s).
const BACKLOG: Duration = Duration::from_millis(500);

/// The counters averaged over `[start, end)`, CUPTI nanoseconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    pub start: u64,
    pub end: u64,
    /// Hz.
    pub sm_clock: f64,
    /// Percent of elapsed cycles an SM had a warp resident, averaged over SMs.
    pub sm_active: f64,
    /// Percent of elapsed cycles the tensor pipes were busy, averaged over SMs.
    pub tensor_active: f64,
    /// Bytes per second.
    pub dram_read: f64,
    /// Bytes per second.
    pub dram_write: f64,
}

/// One device's PM sampling, on from [`Sampler::start`] until dropped.
pub struct Sampler {
    host: *mut cupti::CUpti_Profiler_Host_Object,
    sampling: *mut cupti::CUpti_PmSampling_Object,
    started: bool,
    image: Vec<u8>,
    /// CUPTI reads these through the pointers it was handed for as long as
    /// it samples.
    available: Vec<u8>,
    config: Vec<u8>,
    names: [*const c_char; 5],
    overflows: u64,
    /// Whether the first sample went by: its start is not a timestamp.
    begun: bool,
}

// CUPTI's objects are only ever touched through `&mut self`.
unsafe impl Send for Sampler {}

/// A CUPTI parameter struct, zeroed but for its size and the fields named.
macro_rules! params {
    ($t:ident { $($f:ident: $v:expr),* $(,)? }) => {{
        // SAFETY: CUPTI's parameter structs are plain C: pointers, sizes and enums whose zero is a variant.
        #[allow(unused_unsafe)]
        let mut p: cupti::$t = unsafe { std::mem::zeroed() };
        p.structSize = std::mem::size_of::<cupti::$t>();
        $(p.$f = $v;)*
        p
    }};
}

impl Sampler {
    /// Starts sampling device `device` (a CUDA ordinal) every `interval`.
    pub fn start(device: usize, interval: Duration) -> Result<Sampler> {
        if interval < FLOOR {
            return Err(Error::Cuda(format!(
                "PM sampling every {interval:?}: the host decodes no faster than every {FLOOR:?}"
            )));
        }
        cudarc::driver::result::init().map_err(|e| Error::Cuda(format!("cuInit: {e:?}")))?;
        ok(unsafe { cupti::cuptiProfilerInitialize(&mut params!(CUpti_Profiler_Initialize_Params {})) }, "profiler")?;
        let mut s = Sampler {
            host: std::ptr::null_mut(),
            sampling: std::ptr::null_mut(),
            started: false,
            image: Vec::new(),
            available: Vec::new(),
            config: Vec::new(),
            names: METRICS.map(CStr::as_ptr),
            overflows: 0,
            begun: false,
        };
        let mut chip = params!(CUpti_Device_GetChipName_Params { deviceIndex: device });
        ok(unsafe { cupti::cuptiDeviceGetChipName(&mut chip) }, "chip name")?;
        let mut avail = params!(CUpti_PmSampling_GetCounterAvailability_Params { deviceIndex: device });
        ok(unsafe { cupti::cuptiPmSamplingGetCounterAvailability(&mut avail) }, "counter availability")?;
        s.available = vec![0u8; avail.counterAvailabilityImageSize];
        avail.pCounterAvailabilityImage = s.available.as_mut_ptr();
        ok(unsafe { cupti::cuptiPmSamplingGetCounterAvailability(&mut avail) }, "counter availability")?;
        let mut host = params!(CUpti_Profiler_Host_Initialize_Params {
            profilerType: cupti::CUpti_ProfilerType::CUPTI_PROFILER_TYPE_PM_SAMPLING,
            pChipName: chip.pChipName,
            pCounterAvailabilityImage: s.available.as_ptr(),
        });
        ok(unsafe { cupti::cuptiProfilerHostInitialize(&mut host) }, "profiler host")?;
        s.host = host.pHostObject;
        s.config = s.configure()?;
        let mut enable = params!(CUpti_PmSampling_Enable_Params { deviceIndex: device });
        ok(unsafe { cupti::cuptiPmSamplingEnable(&mut enable) }, "enable PM sampling")?;
        s.sampling = enable.pPmSamplingObject;
        let mut set = params!(CUpti_PmSampling_SetConfig_Params {
            pPmSamplingObject: s.sampling,
            configSize: s.config.len(),
            pConfig: s.config.as_ptr(),
            hardwareBufferSize: HARDWARE_BUFFER,
            samplingInterval: interval.as_nanos() as u64,
            triggerMode: cupti::CUpti_PmSampling_TriggerMode::CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL,
            // The default keeps the oldest, and a full buffer then fails every later decode.
            hwBufferAppendMode:
                cupti::CUpti_PmSampling_HardwareBuffer_AppendMode::CUPTI_PM_SAMPLING_HARDWARE_BUFFER_APPEND_MODE_KEEP_LATEST,
        });
        ok(unsafe { cupti::cuptiPmSamplingSetConfig(&mut set) }, "PM sampling config")?;
        let mut size = params!(CUpti_PmSampling_GetCounterDataSize_Params {
            pPmSamplingObject: s.sampling,
            pMetricNames: s.names.as_mut_ptr(),
            numMetrics: METRICS.len(),
            maxSamples: (BACKLOG.as_nanos() / interval.as_nanos()).clamp(4, u32::MAX.into()) as u32,
        });
        ok(unsafe { cupti::cuptiPmSamplingGetCounterDataSize(&mut size) }, "counter data size")?;
        s.image = vec![0; size.counterDataSize];
        s.clear()?;
        ok(
            unsafe {
                cupti::cuptiPmSamplingStart(&mut params!(CUpti_PmSampling_Start_Params {
                    pPmSamplingObject: s.sampling
                }))
            },
            "start PM sampling",
        )?;
        s.started = true;
        Ok(s)
    }

    /// The configuration image that asks the device for [`METRICS`].
    fn configure(&mut self) -> Result<Vec<u8>> {
        let mut add = params!(CUpti_Profiler_Host_ConfigAddMetrics_Params {
            pHostObject: self.host,
            ppMetricNames: self.names.as_mut_ptr(),
            numMetrics: METRICS.len(),
        });
        ok(unsafe { cupti::cuptiProfilerHostConfigAddMetrics(&mut add) }, "add metrics")?;
        let mut size = params!(CUpti_Profiler_Host_GetConfigImageSize_Params { pHostObject: self.host });
        ok(unsafe { cupti::cuptiProfilerHostGetConfigImageSize(&mut size) }, "config image size")?;
        let mut config = vec![0u8; size.configImageSize];
        let mut get = params!(CUpti_Profiler_Host_GetConfigImage_Params {
            pHostObject: self.host,
            configImageSize: config.len(),
            pConfigImage: config.as_mut_ptr(),
        });
        ok(unsafe { cupti::cuptiProfilerHostGetConfigImage(&mut get) }, "config image")?;
        Ok(config)
    }

    fn clear(&mut self) -> Result<()> {
        let mut init = params!(CUpti_PmSampling_CounterDataImage_Initialize_Params {
            pPmSamplingObject: self.sampling,
            counterDataSize: self.image.len(),
            pCounterData: self.image.as_mut_ptr(),
        });
        ok(unsafe { cupti::cuptiPmSamplingCounterDataImageInitialize(&mut init) }, "counter data image")
    }

    /// The samples completed since the last collect, in time order: all of
    /// them unless more than [`BACKLOG`] waited, and then the rest next time.
    pub fn collect(&mut self) -> Result<Vec<Sample>> {
        let mut decode = params!(CUpti_PmSampling_DecodeData_Params {
            pPmSamplingObject: self.sampling,
            pCounterDataImage: self.image.as_mut_ptr(),
            counterDataImageSize: self.image.len(),
        });
        // The one struct here `size_of` pads past CUPTI's size, which it refuses.
        decode.structSize = std::mem::offset_of!(cupti::CUpti_PmSampling_DecodeData_Params, overflow) + 1;
        ok(unsafe { cupti::cuptiPmSamplingDecodeData(&mut decode) }, "decode PM samples")?;
        self.overflows += u64::from(decode.overflow != 0);
        let mut info = params!(CUpti_PmSampling_GetCounterDataInfo_Params {
            pCounterDataImage: self.image.as_ptr(),
            counterDataImageSize: self.image.len(),
        });
        ok(unsafe { cupti::cuptiPmSamplingGetCounterDataInfo(&mut info) }, "counter data info")?;
        let first = usize::from(!self.begun).min(info.numCompletedSamples);
        self.begun |= info.numCompletedSamples > 0;
        let samples = (first..info.numCompletedSamples).map(|i| self.sample(i)).collect::<Result<Vec<_>>>()?;
        self.clear()?;
        Ok(samples)
    }

    fn sample(&mut self, index: usize) -> Result<Sample> {
        let mut info = params!(CUpti_PmSampling_CounterData_GetSampleInfo_Params {
            pPmSamplingObject: self.sampling,
            pCounterDataImage: self.image.as_ptr(),
            counterDataImageSize: self.image.len(),
            sampleIndex: index,
        });
        ok(unsafe { cupti::cuptiPmSamplingCounterDataGetSampleInfo(&mut info) }, "sample info")?;
        let mut v = [0f64; 5];
        let mut eval = params!(CUpti_Profiler_Host_EvaluateToGpuValues_Params {
            pHostObject: self.host,
            pCounterDataImage: self.image.as_ptr(),
            counterDataImageSize: self.image.len(),
            rangeIndex: index,
            ppMetricNames: self.names.as_mut_ptr(),
            numMetrics: METRICS.len(),
            pMetricValues: v.as_mut_ptr(),
        });
        ok(unsafe { cupti::cuptiProfilerHostEvaluateToGpuValues(&mut eval) }, "evaluate PM sample")?;
        let [sm_clock, sm_active, tensor_active, dram_read, dram_write] = v;
        Ok(Sample {
            start: info.startTimestamp,
            end: info.endTimestamp,
            sm_clock,
            sm_active,
            tensor_active,
            dram_read,
            dram_write,
        })
    }

    /// Decode passes that found the hardware buffer had overwritten records
    /// not yet decoded.
    pub fn overflows(&self) -> u64 {
        self.overflows
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        // Teardown errors have no one left to act on them.
        unsafe {
            if self.started {
                cupti::cuptiPmSamplingStop(&mut params!(CUpti_PmSampling_Stop_Params {
                    pPmSamplingObject: self.sampling
                }));
            }
            if !self.sampling.is_null() {
                cupti::cuptiPmSamplingDisable(&mut params!(CUpti_PmSampling_Disable_Params {
                    pPmSamplingObject: self.sampling
                }));
            }
            if !self.host.is_null() {
                cupti::cuptiProfilerHostDeinitialize(&mut params!(CUpti_Profiler_Host_Deinitialize_Params {
                    pHostObject: self.host
                }));
            }
            cupti::cuptiProfilerDeInitialize(&mut params!(CUpti_Profiler_DeInitialize_Params {}));
        }
    }
}
