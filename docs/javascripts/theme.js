/*
 * Cross-fades the page when the reader switches between light and dark.
 *
 * Material swaps the colour scheme in a single frame, with CSS transitions turned
 * off. Where the browser supports view transitions, the switch runs inside one, so
 * the old page fades into the new. Without that support, or under reduced motion,
 * the switch stays instant.
 */
(() => {
  "use strict";

  const still = window.matchMedia("(prefers-reduced-motion: reduce)");
  let replaying = false;

  /** Resolves once Material has applied the new scheme, or after a short wait. */
  function schemeApplied() {
    return new Promise((resolve) => {
      const observer = new MutationObserver(() => {
        observer.disconnect();
        resolve();
      });
      observer.observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
      setTimeout(() => {
        observer.disconnect();
        resolve();
      }, 300);
    });
  }

  // a click on the toggle, or Material's own click on the next option when Enter is
  // pressed, is held back and replayed inside a view transition
  document.addEventListener(
    "click",
    (event) => {
      if (replaying || !document.startViewTransition || still.matches || !(event.target instanceof Element)) {
        return;
      }
      const control = event.target.closest('[data-md-component="palette"] label, [data-md-component="palette"] input');
      if (!control) {
        return;
      }
      const option = control instanceof HTMLLabelElement ? document.getElementById(control.htmlFor) : control;
      if (!(option instanceof HTMLInputElement) || option.checked) {
        return;
      }
      event.preventDefault();
      document.startViewTransition(() => {
        const applied = schemeApplied();
        replaying = true;
        try {
          option.click();
        } finally {
          replaying = false;
        }
        return applied;
      });
    },
    true
  );
})();
