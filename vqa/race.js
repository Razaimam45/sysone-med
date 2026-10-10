/* Pure replay timing: one sequential worker per model, common elapsed time. */
(function (root) {
  function completionTimes(data, key) {
    let elapsed = 0;
    return Array.from({length: data.total}, (_, i) => {
      const sample = data.samples[i % data.unique];
      elapsed += data.recorded[key][sample.id].latency_ms;
      return elapsed;
    });
  }
  function completedAt(times, elapsed) {
    let low = 0, high = times.length;
    while (low < high) {
      const mid = (low + high) >>> 1;
      if (times[mid] <= elapsed) low = mid + 1;
      else high = mid;
    }
    return low;
  }
  const api = {completionTimes, completedAt};
  if (typeof module !== 'undefined') module.exports = api;
  else root.RaceTiming = api;
})(typeof window === 'undefined' ? globalThis : window);
