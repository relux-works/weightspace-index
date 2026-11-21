const observer = new IntersectionObserver(entries => {
  for (const { target, isIntersecting } of entries) target.toggleAttribute("data-offscreen", !isIntersecting);
});
document.querySelectorAll("[data-crossfade]").forEach(diagram => observer.observe(diagram));

function updatePageVisibility() {
  document.documentElement.toggleAttribute("data-page-hidden", document.hidden);
}
document.addEventListener("visibilitychange", updatePageVisibility);
updatePageVisibility();
