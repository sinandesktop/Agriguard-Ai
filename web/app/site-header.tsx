"use client";

import { useRef, useState } from "react";

function LeafMark() {
  return (
    <svg className="leaf-mark" viewBox="0 0 32 32" aria-hidden="true">
      <path d="M26.5 5.5C16 5.8 8.2 8.8 6.2 16.4c-1.1 4.4 1.7 7.9 5.9 7.7 8.7-.3 12.7-8.3 14.4-18.6Z" />
      <path d="M5 27c4.6-7 9.1-10.9 16.3-14.4" />
    </svg>
  );
}

const LINKS = [
  ["#home", "Home"],
  ["#how-it-works", "How it works"],
  ["#about", "About"],
] as const;

export default function SiteHeader() {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuButton = useRef<HTMLButtonElement>(null);
  const closeMenu = () => setMenuOpen(false);

  return (
    <header className="site-header">
      <div className="site-header-inner">
        <a className="brand" href="#home" aria-label="AgriGuard AI home" onClick={closeMenu}>
          <LeafMark /><span>AgriGuard <b>AI</b></span>
        </a>
        <nav className="site-nav desktop-nav" aria-label="Main navigation">
          {LINKS.map(([href, label]) => <a href={href} key={href}>{label}</a>)}
        </nav>
        <a className="button button-small header-cta" href="#assessment">Check your plant <span aria-hidden="true">↗</span></a>
        <button
          ref={menuButton}
          className="mobile-menu-toggle"
          type="button"
          aria-label={menuOpen ? "Close navigation menu" : "Open navigation menu"}
          aria-expanded={menuOpen}
          aria-controls="mobile-navigation"
          onClick={() => setMenuOpen((open) => !open)}
          onKeyDown={(event) => {
            if (event.key === "Escape") {
              closeMenu();
              menuButton.current?.focus();
            }
          }}
        >
          <span className="menu-icon" aria-hidden="true"><i /><i /><i /></span>
        </button>
        <nav
          id="mobile-navigation"
          className="mobile-nav"
          aria-label="Mobile navigation"
          hidden={!menuOpen}
          onKeyDown={(event) => {
            if (event.key === "Escape") {
              closeMenu();
              menuButton.current?.focus();
            }
          }}
        >
          {LINKS.map(([href, label]) => <a href={href} key={href} onClick={closeMenu}>{label}</a>)}
          <a className="mobile-nav-cta" href="#assessment" onClick={closeMenu}>Check your plant <span aria-hidden="true">↗</span></a>
        </nav>
      </div>
    </header>
  );
}
