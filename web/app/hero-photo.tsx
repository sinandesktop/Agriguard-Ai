"use client";

import { useState } from "react";

export default function HeroPhoto() {
  const [failed, setFailed] = useState(false);

  if (failed) {
    return (
      <div className="hero-photo-fallback" role="img" aria-label="Illustration of a healthy green leaf">
        <svg viewBox="0 0 420 420" aria-hidden="true">
          <path d="M324 83C195 78 96 126 83 225c-8 62 36 106 94 93 105-24 137-127 147-235Z" />
          <path d="M68 355c62-99 119-156 216-216" />
          <path d="M168 251c-4-42-23-65-55-78M218 207c30-15 60-18 92-12M136 287c15-26 35-44 62-56" />
        </svg>
      </div>
    );
  }

  return (
    <img
      className="hero-photo"
      src="/images/hero-tomato-v2.webp"
      alt="Green tomato leaves and fruit growing on the plant"
      width="1600"
      height="1067"
      fetchPriority="high"
      onError={() => setFailed(true)}
    />
  );
}
