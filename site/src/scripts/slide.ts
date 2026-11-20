// A marker that glides to the pressed (aria-pressed) or selected
// (aria-selected) button of a group marked data-slide="thumb" or "line".
// It follows attribute changes, so the scripts that switch the buttons need
// no changes. The first placement does not animate.
function slideMark(group: HTMLElement) {
  const mark = document.createElement("i");
  mark.className = "slide-mark";
  mark.setAttribute("aria-hidden", "true");
  group.prepend(mark);
  group.classList.add("has-mark");
  let ready = false;
  const place = () => {
    const on = [...group.querySelectorAll<HTMLElement>("[aria-pressed='true'], [aria-selected='true']")][0];
    if (!on || !on.offsetWidth) return;
    mark.style.setProperty("--x", `${on.offsetLeft}px`);
    mark.style.setProperty("--y", `${on.offsetTop}px`);
    mark.style.setProperty("--w", `${on.offsetWidth}px`);
    mark.style.setProperty("--h", `${on.offsetHeight}px`);
    if (!ready) {
      ready = true;
      mark.getBoundingClientRect();
      requestAnimationFrame(() => group.classList.add("mark-ready"));
    }
  };
  new MutationObserver(place).observe(group, { subtree: true, attributes: true, attributeFilter: ["aria-pressed", "aria-selected"] });
  new ResizeObserver(place).observe(group);
  place();
}

document.querySelectorAll<HTMLElement>("[data-slide]").forEach(slideMark);
