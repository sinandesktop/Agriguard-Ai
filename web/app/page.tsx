import AssessmentForm from "./assessment/assessment-form";
import HeroPhoto from "./hero-photo";
import SiteHeader from "./site-header";

function LeafMark() {
  return (
    <svg className="leaf-mark" viewBox="0 0 32 32" aria-hidden="true">
      <path d="M26.5 5.5C16 5.8 8.2 8.8 6.2 16.4c-1.1 4.4 1.7 7.9 5.9 7.7 8.7-.3 12.7-8.3 14.4-18.6Z" />
      <path d="M5 27c4.6-7 9.1-10.9 16.3-14.4" />
    </svg>
  );
}

function HeroSection() {
  return (
    <section id="home" className="hero-section" aria-labelledby="hero-title">
      <div className="hero-inner">
        <div className="hero-copy">
          <p className="eyebrow"><span className="eyebrow-dot" />Thoughtful plant care starts here</p>
          <h1 id="hero-title">A healthier plant starts with <em>knowing what to look for.</em></h1>
          <p className="hero-description">Assess leaf symptoms and explore practical, source-backed guidance for your next steps.</p>
          <div className="hero-actions">
            <a className="button" href="#assessment">Check your plant <span aria-hidden="true">↗</span></a>
            <a className="text-link" href="#how-it-works">How it works <span aria-hidden="true">↓</span></a>
          </div>
          <p className="hero-footnote">A preliminary assessment for learning and guidance. Not a confirmed diagnosis.</p>
        </div>
        <div className="hero-visual">
          <HeroPhoto />
          <div className="photo-caption"><span className="caption-mark" />Better questions. More confident next steps.</div>
        </div>
      </div>
    </section>
  );
}

function HowItWorks() {
  const steps = [
    {
      number: "01",
      title: "Choose your plant",
      description: "Select the crop and add a clear photo of the affected leaf.",
      image: "/images/step-potato-leaf.webp",
      alt: "Green potato plant leaves with water droplets",
    },
    {
      number: "02",
      title: "Assess the leaf",
      description: "Use an on-device check for tomato or potato, or choose Groq and agree before sharing a photo online.",
      image: "/images/step-leaf-care.webp",
      alt: "A gardener gently examining a leafy plant outdoors",
    },
    {
      number: "03",
      title: "Explore next steps",
      description: "Review the possible result, available source-backed guidance, and when to ask an expert.",
      image: "/images/step-tomato-vine.webp",
      alt: "Tomatoes ripening on a leafy vine",
    },
  ];
  return (
    <section id="how-it-works" className="content-section how-section" aria-labelledby="how-title">
      <div className="section-heading">
        <p className="eyebrow">A simple place to start</p>
        <h2 id="how-title">From leaf photo to informed next step</h2>
        <p>AgriGuard helps you explore what a leaf may be showing, with clear limits at every step.</p>
      </div>
      <div className="step-grid">
        {steps.map(({ number, title, description, image, alt }) => (
          <article className="step-card" key={number}>
            <img src={image} alt={alt} width="900" height="560" loading="lazy" />
            <div className="step-card-content">
              <span className="step-number">{number}</span>
              <h3>{title}</h3>
              <p>{description}</p>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function TrustSection() {
  return (
    <section id="limits" className="trust-section" aria-labelledby="trust-title">
      <div className="trust-inner">
        <div className="section-heading trust-heading">
          <p className="eyebrow">Clear about what it can do</p>
          <h2 id="trust-title">Useful guidance, with honest limits</h2>
          <p>Plant symptoms can look alike. The result is a starting point for observation and local advice.</p>
        </div>
        <div className="trust-grid">
          <article><span className="trust-index">01</span><h3>Two local crops</h3><p>The on-device model covers tomato and potato only. It always selects one of its six trained classes and may be wrong for unfamiliar plants or conditions.</p></article>
          <article><span className="trust-index">02</span><h3>Guidance where supported</h3><p>Some results have source-backed, non-chemical guidance. Coverage is incomplete; verified chemical products and dosages are not currently offered.</p></article>
          <article><span className="trust-index">03</span><h3>Your photo, your choice</h3><p>Local assessment runs in your browser. Groq receives a photo only after you choose online assessment, agree, and submit. The app does not intentionally save it; provider policies apply.</p></article>
        </div>
      </div>
    </section>
  );
}

function AboutSection() {
  return (
    <section id="about" className="content-section about-section" aria-labelledby="about-title">
      <div className="about-mark"><LeafMark /></div>
      <div>
        <p className="eyebrow">About AgriGuard AI</p>
        <h2 id="about-title">A careful companion for looking closer.</h2>
        <p>AgriGuard AI is an educational plant-image assessment prototype. It combines a traditional machine-learning model for supported crops with an optional online visual assessment service, and offers source-backed guidance where available.</p>
      </div>
    </section>
  );
}

function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="footer-inner">
        <a className="brand footer-brand" href="#home"><LeafMark /><span>AgriGuard <b>AI</b></span></a>
        <p>AI assessment, not a confirmed diagnosis.</p>
        <a className="footer-cta" href="#assessment">Check another plant <span aria-hidden="true">↑</span></a>
        <small>Photos: <a href="https://unsplash.com/@julien_k" target="_blank" rel="noreferrer">Julien Kettmann</a>, <a href="https://unsplash.com/@fukayamamo" target="_blank" rel="noreferrer">Suri Huang</a>, <a href="https://unsplash.com/@jessiemaxwellphotography" target="_blank" rel="noreferrer">Jessie Maxwell</a>, and <a href="https://unsplash.com/@lauraespana" target="_blank" rel="noreferrer">Laura España</a> via Unsplash.</small>
      </div>
    </footer>
  );
}

export default function HomePage() {
  return (
    <>
      <SiteHeader />
      <main className="site-main">
        <HeroSection />
        <AssessmentForm />
        <HowItWorks />
        <TrustSection />
        <AboutSection />
      </main>
      <SiteFooter />
    </>
  );
}
