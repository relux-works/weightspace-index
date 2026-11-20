// Progressive tabs. Markup: [data-tabs] > [role=tablist][hidden] > button[role=tab]
// followed by [data-panel] elements in the same order. Without JavaScript every
// panel shows with its own heading; with it, one panel at a time, arrow keys,
// Home and End move between tabs. Elements with data-tab-select="<label>"
// anywhere on the page select the tab whose data-label matches and scroll it
// into view.
export function enhanceTabs(root: ParentNode = document) {
  for (const group of root.querySelectorAll<HTMLElement>("[data-tabs]:not([data-tabs-ready])")) {
    group.dataset.tabsReady = "";
    const list = group.querySelector<HTMLElement>(":scope > [role=tablist]")!;
    const tabs = [...list.querySelectorAll<HTMLButtonElement>("[role=tab]")];
    const panels = [...group.querySelectorAll<HTMLElement>(":scope > [data-panel]")];
    const uid = group.id || `tabs-${Math.random().toString(36).slice(2, 8)}`;
    const select = (i: number, focus = false) => {
      tabs.forEach((tab, n) => {
        tab.setAttribute("aria-selected", String(i === n));
        tab.tabIndex = i === n ? 0 : -1;
        panels[n].hidden = i !== n;
      });
      if (focus) tabs[i].focus();
      group.dispatchEvent(new CustomEvent("tabchange", { detail: i }));
    };
    tabs.forEach((tab, i) => {
      tab.id ||= `${uid}-tab-${i}`;
      panels[i].id ||= `${uid}-panel-${i}`;
      tab.setAttribute("aria-controls", panels[i].id);
      panels[i].setAttribute("role", "tabpanel");
      panels[i].setAttribute("aria-labelledby", tab.id);
      panels[i].tabIndex = 0;
      tab.addEventListener("click", () => select(i));
      tab.addEventListener("keydown", e => {
        const next = ({ ArrowRight: (i + 1) % tabs.length, ArrowLeft: (i + tabs.length - 1) % tabs.length, Home: 0, End: tabs.length - 1 } as Record<string, number>)[e.key];
        if (next === undefined) return;
        e.preventDefault();
        select(next, true);
      });
    });
    for (const trigger of document.querySelectorAll<HTMLElement>("[data-tab-select]")) {
      trigger.addEventListener("click", () => {
        const target = trigger.dataset.tabSelect;
        const selected = tabs.findIndex(t => t.getAttribute("aria-selected") === "true");
        const i = target === "lora" && tabs[selected]?.dataset.format === "lora" ? selected
          : tabs.findIndex(t => t.dataset.label === target || t.dataset.format === target);
        if (i < 0) return;
        select(i);
        // The setup is the next step after a download: bring it into view if it is not.
        const box = list.getBoundingClientRect();
        if (box.top < 0 || box.bottom > innerHeight) {
          list.scrollIntoView({ block: "start", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
        }
      });
    }
    list.hidden = false;
    group.classList.add("tabs-on");
    select(Math.max(0, tabs.findIndex(t => t.getAttribute("aria-selected") === "true")));
  }
}
enhanceTabs();
