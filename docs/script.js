document.addEventListener('DOMContentLoaded', () => {
  const nav = document.getElementById('nav');
  const navToggle = document.getElementById('navToggle');

  if (navToggle && nav) {
    navToggle.addEventListener('click', () => {
      const isOpen = nav.classList.toggle('open');
      navToggle.setAttribute('aria-expanded', String(isOpen));
    });
    nav.querySelectorAll('a').forEach((a) => {
      a.addEventListener('click', () => {
        if (nav.classList.contains('open')) {
          nav.classList.remove('open');
          navToggle.setAttribute('aria-expanded', 'false');
        }
      });
    });
  }

  const links = Array.from(document.querySelectorAll('.nav a'));
  const sections = links
    .map((a) => document.querySelector(a.getAttribute('href')))
    .filter(Boolean);

  if ('IntersectionObserver' in window && sections.length) {
    const obs = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (!visible) return;
        const id = '#' + visible.target.id;
        links.forEach((a) => a.classList.toggle('active', a.getAttribute('href') === id));
      },
      { threshold: [0.2, 0.35, 0.5, 0.65] }
    );
    sections.forEach((s) => obs.observe(s));
  }

  const copyBtn = document.getElementById('copyBibtex');
  const bibtexBlock = document.getElementById('bibtexBlock');
  const status = document.getElementById('copyStatus');

  if (copyBtn && bibtexBlock) {
    copyBtn.addEventListener('click', async () => {
      const text = bibtexBlock.innerText.trim();
      try {
        await navigator.clipboard.writeText(text);
        if (status) status.textContent = 'Copied!';
        copyBtn.textContent = 'Copied';
        setTimeout(() => {
          copyBtn.textContent = 'Copy BibTeX';
          if (status) status.textContent = '';
        }, 1200);
      } catch {
        if (status) status.textContent = 'Select and copy manually.';
      }
    });
  }
});
