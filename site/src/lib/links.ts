export const NEW_TAB = { target: "_blank", rel: "noopener noreferrer" } as const;

export function externalLinkAttrs(href: string) {
  return /^(?:https?:)?\/\//i.test(href.trim()) ? NEW_TAB : {};
}
