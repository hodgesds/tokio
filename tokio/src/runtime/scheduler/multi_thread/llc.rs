use super::Handle;

use crate::loom::sync::{Arc, Mutex};
use crate::loom::sync::atomic::AtomicUsize;
use crate::runtime::task;
use crate::util::cacheline::CachePadded;

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::atomic::Ordering::{Acquire, Release};

pub(super) const NO_PARTITION: usize = usize::MAX;

/// Per-LLC queues are sharded to avoid a single runtime-wide injection lock.
/// Entries use a small virtual-finish calculation inspired by weighted vtime
/// queues in sched_ext. This is an ordering hint, rather than a CPU-time
/// entitlement.
pub(super) struct LlcQueues {
    partitions: Box<[CachePadded<LlcQueue>]>,
    non_empty: Box<[AtomicUsize]>,
    workers: Box<[AtomicUsize]>,
}

struct LlcQueue {
    len: AtomicUsize,
    state: Mutex<State<task::Notified<Arc<Handle>>>>,
}

struct State<T> {
    closed: bool,
    sequence: u64,
    virtual_time: u64,
    entries: Entries<T>,
}

enum Entries<T> {
    Fifo(VecDeque<T>),
    Weighted(BinaryHeap<Entry<T>>),
}

struct Entry<T> {
    task: T,
    finish: u64,
    sequence: u64,
}

const WEIGHT_SCALE: u64 = 1024 * 1024;
const DEFAULT_WEIGHT: u32 = 1024;

impl LlcQueues {
    pub(super) fn new(partitions: usize) -> Self {
        let partitions: Box<[CachePadded<LlcQueue>]> = (0..partitions)
            .map(|_| CachePadded::new(LlcQueue {
                len: AtomicUsize::new(0),
                state: Mutex::new(State {
                    closed: false,
                    sequence: 0,
                    virtual_time: 0,
                    entries: Entries::Fifo(VecDeque::new()),
                }),
            }))
            .collect();
        let bits = usize::BITS as usize;
        let non_empty = (0..(partitions.len() + bits - 1) / bits)
            .map(|_| AtomicUsize::new(0))
            .collect();
        let workers = (0..partitions.len())
            .map(|_| AtomicUsize::new(0))
            .collect();
        Self {
            partitions,
            non_empty,
            workers,
        }
    }

    pub(super) fn len(&self, partition: usize) -> usize {
        self.partitions[partition].len.load(Acquire)
    }

    pub(super) fn partition_count(&self) -> usize {
        self.partitions.len()
    }

    pub(super) fn worker_count(&self, partition: usize) -> usize {
        self.workers[partition].load(Acquire)
    }

    pub(super) fn update_worker(&self, previous: Option<usize>, next: Option<usize>) {
        if previous == next {
            return;
        }
        if let Some(next) = next {
            self.workers[next].fetch_add(1, Release);
        }
        if let Some(previous) = previous {
            let count = self.workers[previous].fetch_sub(1, Release);
            debug_assert!(count > 0);
        }
    }

    pub(super) fn is_empty(&self, partition: usize) -> bool {
        self.len(partition) == 0
    }

    pub(super) fn all_empty(&self) -> bool {
        self.non_empty.iter().all(|word| word.load(Acquire) == 0)
    }

    pub(super) fn push(
        &self,
        partition: usize,
        task: task::Notified<Arc<Handle>>,
        weight: u32,
    ) {
        let queue = &self.partitions[partition];
        let mut state = queue.state.lock();
        if state.closed {
            return;
        }

        let became_non_empty = state.entries.is_empty();
        state.push(task, weight);
        queue.len.store(state.entries.len(), Release);
        if became_non_empty {
            self.mark_non_empty(partition);
        }
    }

    pub(super) fn pop(
        &self,
        partition: usize,
    ) -> Option<task::Notified<Arc<Handle>>> {
        let queue = &self.partitions[partition];
        if queue.len.load(Acquire) == 0 {
            return None;
        }

        let mut state = queue.state.lock();
        let task = state.pop()?;
        queue.len.store(state.entries.len(), Release);
        if state.entries.is_empty() {
            state.reset_fifo_if_empty();
            self.clear_non_empty(partition);
        }
        Some(task)
    }

    pub(super) fn pop_n<R>(
        &self,
        partition: usize,
        count: usize,
        f: impl FnOnce(Pop<'_>) -> R,
    ) -> R {
        let queue = &self.partitions[partition];
        let mut state = queue.state.lock();
        let count = count.min(state.entries.len());
        let result = f(Pop {
            state: &mut state,
            remaining: count,
        });
        queue.len.store(state.entries.len(), Release);
        if state.entries.is_empty() {
            state.reset_fifo_if_empty();
            self.clear_non_empty(partition);
        }
        result
    }

    /// Pops from a non-empty partition other than `home`, probing at most
    /// `max_probes` queues. The presence bitmap makes the scan proportional to
    /// the number of machine words rather than the number of LLCs.
    pub(super) fn pop_other_where(
        &self,
        home: usize,
        start: usize,
        max_probes: usize,
        mut matches: impl FnMut(usize) -> bool,
    ) -> Option<task::Notified<Arc<Handle>>> {
        let bits = usize::BITS as usize;
        let start_word = (start / bits) % self.non_empty.len();
        let start_bit = start % bits;
        let mut probes = 0;

        for word_offset in 0..self.non_empty.len() {
            let word_index = (start_word + word_offset) % self.non_empty.len();
            let rotation = if word_offset == 0 { start_bit } else { 0 };
            let mut candidates = self.non_empty[word_index].load(Acquire);

            if home / bits == word_index {
                candidates &= !(1 << (home % bits));
            }

            candidates = candidates.rotate_right(rotation as u32);
            while candidates != 0 && probes < max_probes {
                let rotated_bit = candidates.trailing_zeros() as usize;
                candidates &= candidates - 1;
                let bit = (rotated_bit + rotation) % bits;
                let partition = word_index * bits + bit;
                if partition < self.partitions.len() && matches(partition) {
                    probes += 1;
                    if let Some(task) = self.pop(partition) {
                        return Some(task);
                    }
                }
            }

            if probes == max_probes {
                break;
            }
        }
        None
    }

    pub(super) fn close(&self) {
        for queue in &self.partitions {
            queue.state.lock().closed = true;
        }
    }

    #[cfg(all(tokio_unstable, feature = "taskdump"))]
    pub(super) fn drain_into(&self, dst: &mut Vec<task::Notified<Arc<Handle>>>) {
        for (partition, queue) in self.partitions.iter().enumerate() {
            let mut state = queue.state.lock();
            while let Some(task) = state.pop() {
                dst.push(task);
            }
            state.reset_fifo_if_empty();
            queue.len.store(0, Release);
            self.clear_non_empty(partition);
        }
    }

    fn mark_non_empty(&self, partition: usize) {
        let bits = usize::BITS as usize;
        let word = &self.non_empty[partition / bits];
        let bit = 1 << (partition % bits);
        word.fetch_or(bit, Release);
    }

    fn clear_non_empty(&self, partition: usize) {
        let bits = usize::BITS as usize;
        let word = &self.non_empty[partition / bits];
        word.fetch_and(!(1 << (partition % bits)), Release);
    }
}

pub(super) struct Pop<'a> {
    state: &'a mut State<task::Notified<Arc<Handle>>>,
    remaining: usize,
}

impl Iterator for Pop<'_> {
    type Item = task::Notified<Arc<Handle>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        Some(self.state.pop().expect("LLC queue length changed"))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for Pop<'_> {}

impl Drop for Pop<'_> {
    fn drop(&mut self) {
        // Keep the queue's accounting exact even if the consumer intentionally
        // stops early.
        while self.next().is_some() {}
    }
}

impl<T> State<T> {
    fn push(&mut self, task: T, weight: u32) {
        if weight == DEFAULT_WEIGHT {
            if let Entries::Fifo(entries) = &mut self.entries {
                entries.push_back(task);
                return;
            }
        }

        if matches!(self.entries, Entries::Fifo(_)) {
            let Entries::Fifo(entries) = std::mem::replace(
                &mut self.entries,
                Entries::Weighted(BinaryHeap::new()),
            ) else {
                unreachable!();
            };
            let finish = self.finish(DEFAULT_WEIGHT);
            for task in entries {
                let sequence = self.next_sequence();
                let Entries::Weighted(weighted) = &mut self.entries else {
                    unreachable!();
                };
                weighted.push(Entry {
                    task,
                    finish,
                    sequence,
                });
            }
        }

        let finish = self.finish(weight);
        let sequence = self.next_sequence();
        let Entries::Weighted(entries) = &mut self.entries else {
            unreachable!();
        };
        entries.push(Entry {
            task,
            finish,
            sequence,
        });
    }

    fn pop(&mut self) -> Option<T> {
        match &mut self.entries {
            Entries::Fifo(entries) => entries.pop_front(),
            Entries::Weighted(entries) => {
                let entry = entries.pop()?;
                self.virtual_time = self.virtual_time.max(entry.finish);
                Some(entry.task)
            }
        }
    }

    fn finish(&self, weight: u32) -> u64 {
        self.virtual_time
            .saturating_add((WEIGHT_SCALE / u64::from(weight.max(1))).max(1))
    }

    fn next_sequence(&mut self) -> u64 {
        let sequence = self.sequence;
        self.sequence = self.sequence.wrapping_add(1);
        sequence
    }

    fn reset_fifo_if_empty(&mut self) {
        if self.entries.is_empty() && matches!(self.entries, Entries::Weighted(_)) {
            self.entries = Entries::Fifo(VecDeque::new());
            self.virtual_time = 0;
        }
    }
}

impl<T> Entries<T> {
    fn len(&self) -> usize {
        match self {
            Self::Fifo(entries) => entries.len(),
            Self::Weighted(entries) => entries.len(),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Fifo(entries) => entries.is_empty(),
            Self::Weighted(entries) => entries.is_empty(),
        }
    }
}

impl<T> PartialEq for Entry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.finish == other.finish && self.sequence == other.sequence
    }
}

impl<T> Eq for Entry<T> {}

impl<T> PartialOrd for Entry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Entry<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap returns the greatest item. Reverse both comparisons so
        // the lowest virtual finish wins and ties remain FIFO.
        other
            .finish
            .cmp(&self.finish)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

#[cfg(test)]
mod tests {
    use super::{Entries, State, DEFAULT_WEIGHT};

    fn state<T>() -> State<T> {
        State {
            closed: false,
            sequence: 0,
            virtual_time: 0,
            entries: Entries::Fifo(VecDeque::new()),
        }
    }

    use std::collections::VecDeque;

    #[test]
    fn default_weights_stay_fifo() {
        let mut state = state();
        state.push('a', DEFAULT_WEIGHT);
        state.push('b', DEFAULT_WEIGHT);
        state.push('c', DEFAULT_WEIGHT);

        assert!(matches!(&state.entries, Entries::Fifo(_)));
        assert_eq!(state.pop(), Some('a'));
        assert_eq!(state.pop(), Some('b'));
        assert_eq!(state.pop(), Some('c'));
    }

    #[test]
    fn non_default_weight_promotes_fifo_without_reordering_ties() {
        let mut state = state();
        state.push('a', DEFAULT_WEIGHT);
        state.push('b', DEFAULT_WEIGHT);
        state.push('c', DEFAULT_WEIGHT * 2);

        assert!(matches!(&state.entries, Entries::Weighted(_)));
        assert_eq!(state.pop(), Some('c'));
        assert_eq!(state.pop(), Some('a'));
        assert_eq!(state.pop(), Some('b'));
    }

    #[test]
    fn equal_non_default_weights_are_fifo() {
        let mut state = state();
        state.push('a', DEFAULT_WEIGHT / 2);
        state.push('b', DEFAULT_WEIGHT / 2);
        state.push('c', DEFAULT_WEIGHT / 2);

        assert_eq!(state.pop(), Some('a'));
        assert_eq!(state.pop(), Some('b'));
        assert_eq!(state.pop(), Some('c'));
    }

    #[test]
    fn drained_weighted_queue_returns_to_fifo() {
        let mut state = state();
        state.push('a', DEFAULT_WEIGHT * 2);
        assert_eq!(state.pop(), Some('a'));
        state.reset_fifo_if_empty();

        assert!(matches!(&state.entries, Entries::Fifo(_)));
        assert_eq!(state.virtual_time, 0);
        state.push('b', DEFAULT_WEIGHT);
        assert!(matches!(&state.entries, Entries::Fifo(_)));
    }

    #[test]
    fn edge_weights_are_clamped_and_ordered() {
        let mut state = state();
        state.push("zero", 0);
        state.push("one", 1);
        state.push("max", u32::MAX);

        assert_eq!(state.pop(), Some("max"));
        assert_eq!(state.pop(), Some("zero"));
        assert_eq!(state.pop(), Some("one"));
    }

    #[test]
    fn virtual_time_saturation_preserves_fifo_ties() {
        let mut state = state();
        state.virtual_time = u64::MAX;
        state.push('a', 1);
        state.push('b', u32::MAX);

        assert_eq!(state.pop(), Some('a'));
        assert_eq!(state.pop(), Some('b'));
        assert_eq!(state.virtual_time, u64::MAX);
    }

    #[test]
    fn recurring_tasks_make_weighted_progress_without_starvation() {
        let mut state = state();
        for task in [(0, 512), (1, 1024), (2, 2048)] {
            state.push(task, task.1);
        }

        let mut polls = [0; 3];
        for _ in 0..700 {
            let task = state.pop().unwrap();
            polls[task.0] += 1;
            state.push(task, task.1);
        }

        assert_eq!(polls, [100, 200, 400]);
    }
}
