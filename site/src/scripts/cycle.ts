// Auto-advance for a row of step buttons. Each button may set data-ms for how
// long its step stays up (default 5 s). Hover, focus, scrolling away and a
// hidden tab pause the timer where it is; a click hands control to the reader
// and stops the cycle. Reduced motion never starts it.
export function autoCycle(root: HTMLElement, buttons: HTMLElement[], go: (i: number) => void) {
  if (buttons.length < 2 || matchMedia("(prefers-reduced-motion: reduce)").matches) return () => {};
  const ms = (i: number) => Number(buttons[i].dataset.ms) || 5000;
  let i = Math.max(0, buttons.findIndex(b => b.getAttribute("aria-pressed") === "true"));
  let left = ms(i), since = 0, timer = 0, on = true;
  let hover = false, focus = false, offscreen = false;
  const paused = () => hover || focus || offscreen || document.hidden;
  const pause = () => {
    if (!timer) return;
    clearTimeout(timer);
    timer = 0;
    left = Math.max(400, left - (performance.now() - since));
  };
  const run = () => {
    if (!on || timer || paused()) return;
    since = performance.now();
    timer = window.setTimeout(() => {
      timer = 0;
      i = (i + 1) % buttons.length;
      go(i);
      left = ms(i);
      run();
    }, left);
  };
  const update = () => (paused() ? pause() : run());
  const seen = new IntersectionObserver(([e]) => { offscreen = !e.isIntersecting; update(); }, { threshold: 0.35 });
  seen.observe(root);
  root.addEventListener("pointerenter", () => { hover = true; update(); });
  root.addEventListener("pointerleave", () => { hover = false; update(); });
  root.addEventListener("focusin", () => { focus = true; update(); });
  root.addEventListener("focusout", e => { focus = root.contains(e.relatedTarget as Node); update(); });
  document.addEventListener("visibilitychange", update);
  return () => {
    on = false;
    pause();
    seen.disconnect();
    document.removeEventListener("visibilitychange", update);
  };
}
