"""Console progress reporting shared by every task.

Three tools, and nothing else in the project prints progress:

  step(label)        a timed block: prints a start line and a done line, and while
                     it runs a background heartbeat prints "still running" every
                     HEARTBEAT_SECONDS, so a long fit never looks like a hang.
  step(label, parallel_tasks=n)
                     the same, but it also counts the joblib tasks (e.g. the CV fits
                     of a RandomizedSearchCV) that finish inside the block and shows
                     done/total and an ETA in the heartbeat.
  track(items, label)
                     a loop over items where each item runs inside its own step, with
                     an [i/n] counter and an ETA for the rest of the loop.

Typical output:
  [14:02:01] >> Tuning [5/6] RandomForest   (ETA for the rest ~3m)
  [14:02:01]   >> CV search: 8 candidates x 5 folds
  [14:02:31]   .. still running: Tuning [5/6] RandomForest > CV search ... | 1/40 tasks | 30s | ETA ~19m30s
  [14:22:07]   << CV search: 8 candidates x 5 folds - done in 20m06s
"""
import sys
import threading
import time
from contextlib import contextmanager

import joblib

import config

# Output piped to a file (`python main.py > out.txt`, `| tee`) is block-buffered
# by default, so progress would only show up in chunks of several KB. Flush on
# every newline instead. (Notebook streams have no reconfigure and are unbuffered.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

RUN_START = time.perf_counter()

_stack = []                   # active steps, outermost first
_lock = threading.RLock()     # keeps heartbeat lines from interleaving with step lines
_last_output = time.perf_counter()
_heartbeat = None


def fmt_duration(seconds):
    """Turn 4012.3 into '1h06m52s', for elapsed times and ETAs."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s" if m else f"{s}s"


def run_elapsed():
    """Time since the program started, shown in the task banners."""
    return fmt_duration(time.perf_counter() - RUN_START)


def log(message, depth=None):
    """Timestamped line, indented to the nesting depth of the current step."""
    global _last_output
    depth = len(_stack) if depth is None else depth
    with _lock:
        print(f"[{time.strftime('%H:%M:%S')}] {'  ' * depth}{message}", flush=True)
        _last_output = time.perf_counter()


class _Step:
    def __init__(self, label, parallel_tasks):
        self.label = label
        self.total = parallel_tasks
        self.done = 0
        self.owner = None     # the joblib.Parallel whose tasks this step counts
        self.start = time.perf_counter()

    def elapsed(self):
        return time.perf_counter() - self.start

    def status(self):
        """'3/40 tasks | 4m10s | ETA ~51m' - the counter part only when counting."""
        parts = []
        if self.total:
            parts.append(f"{self.done}/{self.total} tasks")
        parts.append(fmt_duration(self.elapsed()))
        if self.total and 0 < self.done < self.total:
            eta = self.elapsed() / self.done * (self.total - self.done)
            parts.append(f"ETA ~{fmt_duration(eta)}")
        elif self.total and self.done >= self.total:
            parts.append("all tasks done, finishing up (e.g. refitting the best candidate)")
        return " | ".join(parts)


@contextmanager
def step(label, parallel_tasks=None, note=""):
    """Time a block of work and keep the console alive while it runs.

    `parallel_tasks` is the number of joblib tasks the block will run (CV fits,
    permutation columns, ...); when given, the first joblib.Parallel started
    inside the block that is not a model's own internal threading is counted
    (see the joblib section below). `note` is shown on the start line only.
    """
    _ensure_heartbeat()
    depth = len(_stack)
    log(f">> {label}" + (f"   ({note})" if note else ""), depth)
    s = _Step(label, parallel_tasks)
    _stack.append(s)
    try:
        yield s
    except Exception:
        log(f"!! {label} - FAILED after {fmt_duration(s.elapsed())}", depth)
        raise
    else:
        log(f"<< {label} - done in {fmt_duration(s.elapsed())}", depth)
    finally:
        _stack.remove(s)


def _describe(item):
    """Default log name: a (name, object, ...) tuple is named by its first element."""
    return str(item[0] if isinstance(item, tuple) else item)


def track(items, label, describe=_describe):
    """Iterate over `items`, running each one inside its own step.

    `describe(item)` names the item in the log; by default a tuple such as the
    (name, pipeline) pairs of the ranked model list is named by its first element,
    anything else by str(item). The start line of
    each item after the first carries an ETA for the rest of the loop, based on
    the mean time per item so far, so it is rough when items differ in cost.
    """
    items = list(items)
    n = len(items)
    loop_start = time.perf_counter()
    for i, item in enumerate(items, start=1):
        note = ""
        if i > 1:
            per_item = (time.perf_counter() - loop_start) / (i - 1)
            note = f"ETA for the rest ~{fmt_duration(per_item * (n - i + 1))}"
        with step(f"{label} [{i}/{n}] {describe(item)}", note=note):
            yield item


# ----------------------------------------------------------------------
# Heartbeat
# ----------------------------------------------------------------------
def _ensure_heartbeat():
    """Start the background heartbeat thread once per process."""
    global _heartbeat
    if _heartbeat is None:
        _heartbeat = threading.Thread(target=_heartbeat_loop, name="progress-heartbeat",
                                      daemon=True)
        _heartbeat.start()


def _heartbeat_loop():
    """Print 'still running' when nothing has been printed for HEARTBEAT_SECONDS."""
    interval = config.HEARTBEAT_SECONDS
    while True:
        time.sleep(min(5, interval))
        with _lock:
            if not _stack or time.perf_counter() - _last_output < interval:
                continue
            path = " > ".join(s.label for s in _stack)
            counted = next((s for s in reversed(_stack) if s.total), _stack[-1])
            log(f".. still running: {path} | {counted.status()}")


# ----------------------------------------------------------------------
# joblib task counting
# ----------------------------------------------------------------------
# joblib calls Parallel.print_progress() in the parent process after every
# finished task, for both the sequential and the multi-process backends, with
# n_completed_tasks already updated. Wrapping it is the only way to see inside a
# RandomizedSearchCV or permutation_importance while it runs. The wrappers only
# act while a step with `parallel_tasks` is active; otherwise they pass through.
#
# Which Parallel gets counted: a model's own parallelism (the trees of a
# RandomForest, the neighbour chunks of kNN and SMOTE) always asks joblib for
# threads or shared memory, while the loops we want to count (CV fits,
# permutation columns) do not. So the first Parallel started inside the step
# that asked for neither claims the counter. Without this filter the forest's
# 400-tree loop during the final refit, or during permutation_importance's
# baseline predict, would overwrite the fit counter.
def _counting_step():
    return next((s for s in reversed(_stack) if s.total), None)


def _wrap_parallel():
    if getattr(joblib.Parallel.print_progress, "_progress_wrapped", False):
        return    # module re-imported (importlib.reload in a notebook): wrap once only
    original_init = joblib.Parallel.__init__
    original_call = joblib.Parallel.__call__
    original_progress = joblib.Parallel.print_progress

    def __init__(self, *args, **kwargs):
        # Read, never inject: joblib's own defaults for these are sentinels.
        self._model_internal = (kwargs.get("prefer") == "threads"
                                or kwargs.get("require") == "sharedmem")
        original_init(self, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        s = _counting_step()
        if s is not None and s.owner is None and not self._model_internal:
            s.owner = self
        return original_call(self, *args, **kwargs)

    def print_progress(self):
        s = _counting_step()
        if s is not None and s.owner is self:
            s.done = self.n_completed_tasks
        return original_progress(self)

    print_progress._progress_wrapped = True
    joblib.Parallel.__init__ = __init__
    joblib.Parallel.__call__ = __call__
    joblib.Parallel.print_progress = print_progress


_wrap_parallel()
