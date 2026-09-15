//! A/B benchmarks for the LLC-aware multi-thread scheduler.
//!
//! Run with:
//! `RUSTFLAGS="--cfg tokio_unstable" cargo bench --manifest-path benches/Cargo.toml --bench rt_llc_aware`
//!
//! `TOKIO_LLC_BENCH_WORKERS` overrides the worker count. By default, the
//! benchmark uses at least one worker per discovered LLC so the enabled case
//! cannot silently take the undersized-pool fallback.
//! `TOKIO_LLC_BENCH_TASKS` overrides the number of tasks in the spawn and wake
//! benchmarks. The default is 4,096.
//!
//! `TOKIO_LLC_BENCH_CACHE_BYTES` overrides the per-task working-set size in
//! the cache-affine wake benchmarks. The default is 2 MiB per task, with two
//! tasks per LLC. Workers are pinned across the discovered LLCs so each task
//! first-touches its allocation on its target LLC and worker placement remains
//! consistent between enabled and disabled runs.
//! `TOKIO_LLC_BENCH_CACHE_TASKS_PER_LLC` can increase the task density to fill
//! or exceed each LLC; for example, 16 tasks use 32 MiB per LLC by default.

#[cfg(all(tokio_unstable, target_os = "linux"))]
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
#[cfg(all(tokio_unstable, target_os = "linux"))]
use std::collections::BTreeMap;
#[cfg(all(tokio_unstable, target_os = "linux"))]
use std::future::poll_fn;
#[cfg(all(tokio_unstable, target_os = "linux"))]
use std::io;
#[cfg(all(tokio_unstable, target_os = "linux"))]
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc, Arc,
};
#[cfg(all(tokio_unstable, target_os = "linux"))]
use std::task::{Poll, Waker};
#[cfg(all(tokio_unstable, target_os = "linux"))]
use std::time::{Duration, Instant};
#[cfg(all(tokio_unstable, target_os = "linux"))]
use tokio::runtime::{Builder, LlcAwareConfig, Runtime};

#[cfg(all(tokio_unstable, target_os = "linux"))]
const DEFAULT_TASKS: usize = 4_096;

#[cfg(all(tokio_unstable, target_os = "linux"))]
const CACHE_BYTES_PER_TASK: usize = 2 * 1024 * 1024;

#[cfg(all(tokio_unstable, target_os = "linux"))]
const CACHE_TASKS_PER_PARTITION: usize = 2;

#[cfg(all(tokio_unstable, target_os = "linux"))]
const CACHE_ROUNDS: usize = 4;

#[cfg(all(tokio_unstable, target_os = "linux"))]
const WEIGHTED_CACHE_TASKS_PER_PARTITION: usize = 6;

#[cfg(all(tokio_unstable, target_os = "linux"))]
const WEIGHTED_CACHE_WEIGHTS: &[u32] = &[512, 1024, 2048];

#[cfg(all(tokio_unstable, target_os = "linux"))]
#[derive(Clone, Copy)]
enum Mode {
    Disabled,
    Enabled,
}

#[cfg(all(tokio_unstable, target_os = "linux"))]
impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Enabled => "enabled",
        }
    }
}

#[cfg(all(tokio_unstable, target_os = "linux"))]
fn remote_spawn(c: &mut Criterion) {
    let (workers, topology) = benchmark_topology();
    let worker_cpus = benchmark_worker_cpus(workers, topology.partition_count())
        .expect("the LLC benchmark requires Linux sysfs topology");
    let tasks = benchmark_tasks();
    let mut group = c.benchmark_group(format!(
        "llc_aware/remote_spawn/{workers}_workers/{tasks}_tasks"
    ));
    group.throughput(Throughput::Elements(tasks as u64));

    for mode in [Mode::Disabled, Mode::Enabled] {
        let runtime = pinned_runtime(mode, workers, topology.clone(), worker_cpus.clone());
        group.bench_with_input(BenchmarkId::from_parameter(mode.name()), &mode, |b, _| {
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                let mut handles = Vec::with_capacity(tasks);

                for _ in 0..iterations {
                    let start = Instant::now();
                    for _ in 0..tasks {
                        handles.push(runtime.spawn(async {}));
                    }
                    runtime.block_on(async {
                        for handle in handles.drain(..) {
                            handle.await.unwrap();
                        }
                    });
                    elapsed += start.elapsed();
                }

                elapsed
            });
        });
    }

    group.finish();
}

/// Measures explicit per-task placement and weighting through `task::Builder`.
#[cfg(all(tokio_unstable, target_os = "linux"))]
fn hinted_spawn(c: &mut Criterion) {
    let (workers, topology) = benchmark_topology();
    let worker_cpus = benchmark_worker_cpus(workers, topology.partition_count())
        .expect("the LLC benchmark requires Linux sysfs topology");
    let tasks = benchmark_tasks();
    let partitions = topology.partition_count();
    let mut group = c.benchmark_group(format!(
        "llc_aware/hinted_spawn/{workers}_workers/{tasks}_tasks"
    ));
    group.throughput(Throughput::Elements(tasks as u64));

    for mode in [Mode::Disabled, Mode::Enabled] {
        let runtime = pinned_runtime(mode, workers, topology.clone(), worker_cpus.clone());
        group.bench_with_input(BenchmarkId::from_parameter(mode.name()), &mode, |b, _| {
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                let mut handles = Vec::with_capacity(tasks);

                for _ in 0..iterations {
                    let start = Instant::now();
                    for task in 0..tasks {
                        let lane = task / partitions;
                        handles.push(
                            tokio::task::Builder::new()
                                .llc_partition(task % partitions)
                                .weight(512 << (lane % 3))
                                .spawn_on(async {}, runtime.handle())
                                .unwrap(),
                        );
                    }
                    runtime.block_on(async {
                        for handle in handles.drain(..) {
                            handle.await.unwrap();
                        }
                    });
                    elapsed += start.elapsed();
                }

                elapsed
            });
        });
    }

    group.finish();
}

/// Measures remote wakeups of tasks which have already run on a worker. The
/// enabled case therefore sends them through their last-observed LLC queue;
/// the disabled case sends the same wakeups through the global injection queue.
#[cfg(all(tokio_unstable, target_os = "linux"))]
fn affine_wake(c: &mut Criterion) {
    let (workers, topology) = benchmark_topology();
    let worker_cpus = benchmark_worker_cpus(workers, topology.partition_count())
        .expect("the LLC benchmark requires Linux sysfs topology");
    let tasks = benchmark_tasks();
    let mut group = c.benchmark_group(format!(
        "llc_aware/affine_wake/{workers}_workers/{tasks}_tasks"
    ));
    group.throughput(Throughput::Elements(tasks as u64));

    for mode in [Mode::Disabled, Mode::Enabled] {
        let runtime = pinned_runtime(mode, workers, topology.clone(), worker_cpus.clone());
        group.bench_with_input(BenchmarkId::from_parameter(mode.name()), &mode, |b, _| {
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;

                for _ in 0..iterations {
                    let (ready_tx, ready_rx) = mpsc::channel();
                    let mut wakes = Vec::with_capacity(tasks);
                    let mut handles = Vec::with_capacity(tasks);

                    for _ in 0..tasks {
                        let ready_tx = ready_tx.clone();
                        let (wake_tx, wake_rx) = tokio::sync::oneshot::channel();
                        wakes.push(wake_tx);
                        handles.push(runtime.spawn(async move {
                            ready_tx.send(()).unwrap();
                            wake_rx.await.unwrap();
                        }));
                    }
                    drop(ready_tx);
                    for _ in 0..tasks {
                        ready_rx.recv().unwrap();
                    }

                    let start = Instant::now();
                    for wake in wakes {
                        wake.send(()).unwrap();
                    }
                    runtime.block_on(async {
                        for handle in handles {
                            handle.await.unwrap();
                        }
                    });
                    elapsed += start.elapsed();
                }

                elapsed
            });
        });
    }

    group.finish();
}

/// Measures whether a remotely woken task returns to the LLC containing its
/// working set. Every task repeatedly traverses a randomized pointer cycle,
/// making each load depend on the preceding load and defeating sequential
/// prefetching. The configured working set is larger than typical private
/// caches but small enough for multiple tasks to remain resident in an LLC.
#[cfg(all(tokio_unstable, target_os = "linux"))]
fn cache_affine_wake(c: &mut Criterion) {
    cache_affine_wake_with_weights(c, CACHE_TASKS_PER_PARTITION, None);
}

/// Runs the cache-affine workload with six tasks and three weights in every LLC,
/// exercising the weighted queue without changing the memory access pattern.
#[cfg(all(tokio_unstable, target_os = "linux"))]
fn weighted_cache_affine_wake(c: &mut Criterion) {
    cache_affine_wake_with_weights(
        c,
        WEIGHTED_CACHE_TASKS_PER_PARTITION,
        Some(WEIGHTED_CACHE_WEIGHTS),
    );
}

#[cfg(all(tokio_unstable, target_os = "linux"))]
fn cache_affine_wake_with_weights(
    c: &mut Criterion,
    default_tasks_per_partition: usize,
    weights: Option<&[u32]>,
) {
    let (workers, topology) = benchmark_topology();
    let partitions = topology.partition_count();
    let cache_bytes = std::env::var("TOKIO_LLC_BENCH_CACHE_BYTES")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|bytes| *bytes >= 2 * std::mem::size_of::<usize>())
        .unwrap_or(CACHE_BYTES_PER_TASK);
    let tasks_per_partition = std::env::var("TOKIO_LLC_BENCH_CACHE_TASKS_PER_LLC")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|tasks| *tasks > 0)
        .unwrap_or(default_tasks_per_partition);
    let tasks = partitions * tasks_per_partition;
    let bytes_touched = tasks * cache_bytes * CACHE_ROUNDS;
    let worker_cpus = benchmark_worker_cpus(workers, partitions)
        .expect("the cache benchmark requires Linux sysfs topology");
    let workload = if weights.is_some() {
        "cache_affine_wake_weighted"
    } else {
        "cache_affine_wake"
    };
    let mut group = c.benchmark_group(format!(
        "llc_aware/{workload}/{workers}_workers/{cache_bytes}_bytes/{tasks_per_partition}_tasks_per_llc"
    ));
    group.throughput(Throughput::Bytes(bytes_touched as u64));

    for mode in [Mode::Disabled, Mode::Enabled] {
        let runtime = pinned_runtime(mode, workers, topology.clone(), worker_cpus.clone());
        group.bench_with_input(BenchmarkId::from_parameter(mode.name()), &mode, |b, _| {
            let (ready_tx, ready_rx) = mpsc::channel::<Waker>();
            let stop = Arc::new(AtomicBool::new(false));
            let mut handles = Vec::with_capacity(tasks);

            for task in 0..tasks {
                let ready_tx = ready_tx.clone();
                let stop = stop.clone();
                let partition = task % partitions;
                let mut builder = tokio::task::Builder::new().llc_partition(partition);
                if let Some(weights) = weights {
                    let lane = task / partitions;
                    builder = builder.weight(weights[lane % weights.len()]);
                }
                handles.push(
                    builder
                        .spawn_on(
                            async move {
                                // Allocation and first touch happen on the target worker
                                // before timing begins.
                                let links = pointer_cycle(cache_bytes, task as u64);
                                let mut cursor = chase_pointers(&links, task % links.len());

                                loop {
                                    wait_for_external_wake(&ready_tx).await;
                                    if stop.load(Ordering::Acquire) {
                                        break;
                                    }
                                    cursor = chase_pointers(&links, cursor);
                                }

                                (links, black_box(cursor))
                            },
                            runtime.handle(),
                        )
                        .unwrap(),
                );
            }
            drop(ready_tx);

            // Every task has initialized and reached its pending point before
            // Criterion starts measuring the steady-state wake workload.
            let mut wakers = receive_wakers(&ready_rx, tasks);
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;

                for _ in 0..iterations {
                    let start = Instant::now();
                    for _ in 0..CACHE_ROUNDS {
                        for waker in wakers.drain(..) {
                            waker.wake();
                        }
                        wakers = receive_wakers(&ready_rx, tasks);
                    }
                    elapsed += start.elapsed();
                }

                elapsed
            });

            stop.store(true, Ordering::Release);
            for waker in wakers {
                waker.wake();
            }
            let completed = runtime.block_on(async {
                let mut completed = Vec::with_capacity(tasks);
                for handle in handles {
                    completed.push(handle.await.unwrap());
                }
                completed
            });
            drop(black_box(completed));
        });
    }

    group.finish();
}

#[cfg(all(tokio_unstable, target_os = "linux"))]
async fn wait_for_external_wake(ready: &mpsc::Sender<Waker>) {
    let mut registered = false;
    poll_fn(|cx| {
        if registered {
            Poll::Ready(())
        } else {
            registered = true;
            ready.send(cx.waker().clone()).unwrap();
            Poll::Pending
        }
    })
    .await
}

#[cfg(all(tokio_unstable, target_os = "linux"))]
fn receive_wakers(ready: &mpsc::Receiver<Waker>, tasks: usize) -> Vec<Waker> {
    (0..tasks).map(|_| ready.recv().unwrap()).collect()
}

#[cfg(all(tokio_unstable, target_os = "linux"))]
fn pointer_cycle(bytes: usize, seed: u64) -> Vec<usize> {
    let words = (bytes / std::mem::size_of::<usize>()).max(2);
    let mut order = (0..words).collect::<Vec<_>>();
    let mut random = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);

    for index in (1..words).rev() {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        order.swap(index, random as usize % (index + 1));
    }

    let mut links = vec![0; words];
    for pair in order.windows(2) {
        links[pair[0]] = pair[1];
    }
    links[*order.last().unwrap()] = order[0];
    links
}

#[cfg(all(tokio_unstable, target_os = "linux"))]
fn chase_pointers(links: &[usize], mut cursor: usize) -> usize {
    for _ in 0..links.len() {
        cursor = links[cursor];
    }
    black_box(cursor)
}

#[cfg(all(tokio_unstable, target_os = "linux"))]
fn benchmark_topology() -> (usize, LlcAwareConfig) {
    let topology = LlcAwareConfig::from_linux_topology()
        .expect("the LLC benchmark requires Linux sysfs topology");
    let minimum = topology.partition_count().max(2);
    let workers = std::env::var("TOKIO_LLC_BENCH_WORKERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(minimum);
    assert!(
        workers >= topology.partition_count(),
        "TOKIO_LLC_BENCH_WORKERS={workers} is smaller than the {} discovered LLCs",
        topology.partition_count(),
    );
    (workers, topology)
}

#[cfg(all(tokio_unstable, target_os = "linux"))]
fn benchmark_tasks() -> usize {
    std::env::var("TOKIO_LLC_BENCH_TASKS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|tasks| *tasks > 0)
        .unwrap_or(DEFAULT_TASKS)
}

#[cfg(all(tokio_unstable, target_os = "linux"))]
fn pinned_runtime(
    mode: Mode,
    workers: usize,
    topology: LlcAwareConfig,
    worker_cpus: Arc<[usize]>,
) -> Runtime {
    let next_worker = Arc::new(AtomicUsize::new(0));
    let mut builder = Builder::new_multi_thread();
    builder.worker_threads(workers).enable_all();
    builder.on_thread_start(move || {
        let worker = next_worker.fetch_add(1, Ordering::Relaxed);
        if let Some(&cpu) = worker_cpus.get(worker) {
            pin_current_thread(cpu);
        }
    });
    match mode {
        Mode::Disabled => {
            builder.disable_llc_aware();
        }
        Mode::Enabled => {
            builder.llc_aware(topology);
        }
    }
    builder.build().unwrap()
}

#[cfg(all(tokio_unstable, target_os = "linux"))]
fn benchmark_worker_cpus(workers: usize, partitions: usize) -> io::Result<Arc<[usize]>> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    let allowed = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "allowed CPU list is missing"))?;
    let mut cpus_by_llc = BTreeMap::<String, Vec<usize>>::new();
    for cpu in parse_cpu_list(allowed.trim())? {
        cpus_by_llc.entry(llc_key(cpu)?).or_default().push(cpu);
    }
    if cpus_by_llc.len() != partitions {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "found {} LLC CPU groups for {partitions} partitions",
                cpus_by_llc.len()
            ),
        ));
    }

    // Select CPUs round-robin by LLC so the first `partitions` workers place
    // exactly one worker in every LLC. Additional workers remain balanced.
    let mut selected = Vec::with_capacity(workers);
    let mut offset = 0;
    while selected.len() < workers {
        let previous_len = selected.len();
        for cpus in cpus_by_llc.values() {
            if let Some(&cpu) = cpus.get(offset) {
                selected.push(cpu);
                if selected.len() == workers {
                    break;
                }
            }
        }
        if selected.len() == previous_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{workers} workers exceed the allowed CPU count"),
            ));
        }
        offset += 1;
    }
    Ok(selected.into())
}

#[cfg(all(tokio_unstable, target_os = "linux"))]
fn pin_current_thread(cpu: usize) {
    assert!(
        cpu < libc::CPU_SETSIZE as usize,
        "CPU {cpu} exceeds cpu_set_t capacity"
    );
    // SAFETY: `cpu_set_t` is initialized before use, `CPU_SET` receives a CPU
    // found in this process's allowed CPU set, and pthread_self is valid for
    // the duration of pthread_setaffinity_np.
    let result = unsafe {
        let mut set = std::mem::zeroed::<libc::cpu_set_t>();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        libc::pthread_setaffinity_np(
            libc::pthread_self(),
            std::mem::size_of::<libc::cpu_set_t>(),
            &set,
        )
    };
    assert_eq!(result, 0, "failed to pin runtime worker to CPU {cpu}");
}

#[cfg(all(tokio_unstable, target_os = "linux"))]
fn llc_key(cpu: usize) -> io::Result<String> {
    let cache_dir = format!("/sys/devices/system/cpu/cpu{cpu}/cache");
    let mut best = None::<(u32, String)>;

    for entry in std::fs::read_dir(cache_dir)? {
        let path = entry?.path();
        let Ok(level) = read_trimmed(path.join("level")).and_then(|level| {
            level
                .parse::<u32>()
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
        }) else {
            continue;
        };
        let Ok(kind) = read_trimmed(path.join("type")) else {
            continue;
        };
        if kind != "Unified" && kind != "Data" {
            continue;
        }

        let Ok(identity) =
            read_trimmed(path.join("id")).or_else(|_| read_trimmed(path.join("shared_cpu_list")))
        else {
            continue;
        };
        if best
            .as_ref()
            .map_or(true, |(best_level, _)| level > *best_level)
        {
            best = Some((level, format!("{level}:{identity}")));
        }
    }

    if let Some((_, key)) = best {
        return Ok(key);
    }

    read_trimmed(format!(
        "/sys/devices/system/cpu/cpu{cpu}/topology/llc_id"
    ))
}

#[cfg(all(tokio_unstable, target_os = "linux"))]
fn read_trimmed(path: impl AsRef<std::path::Path>) -> io::Result<String> {
    Ok(std::fs::read_to_string(path)?.trim().to_owned())
}

#[cfg(all(tokio_unstable, target_os = "linux"))]
fn parse_cpu_list(list: &str) -> io::Result<Vec<usize>> {
    let mut cpus = Vec::new();
    for range in list.split(',') {
        let (start, end) = match range.split_once('-') {
            Some((start, end)) => (parse_cpu(start)?, parse_cpu(end)?),
            None => {
                let cpu = parse_cpu(range)?;
                (cpu, cpu)
            }
        };
        if start > end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid CPU range",
            ));
        }
        cpus.extend(start..=end);
    }
    Ok(cpus)
}

#[cfg(all(tokio_unstable, target_os = "linux"))]
fn parse_cpu(value: &str) -> io::Result<usize> {
    value
        .parse()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

#[cfg(all(tokio_unstable, target_os = "linux"))]
criterion_group!(
    llc_aware_benches,
    remote_spawn,
    hinted_spawn,
    affine_wake,
    cache_affine_wake,
    weighted_cache_affine_wake
);
#[cfg(all(tokio_unstable, target_os = "linux"))]
criterion_main!(llc_aware_benches);

#[cfg(not(all(tokio_unstable, target_os = "linux")))]
fn main() {
    eprintln!("rt_llc_aware requires Linux and RUSTFLAGS=\"--cfg tokio_unstable\"");
}
