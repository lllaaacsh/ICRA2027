const header = document.querySelector("[data-header]");
const navLinks = [...document.querySelectorAll("nav a")];
const sections = navLinks
  .map((link) => document.querySelector(link.getAttribute("href")))
  .filter(Boolean);

function updateHeader() {
  header?.classList.toggle("scrolled", window.scrollY > 24);
}

function updateActiveSection() {
  const marker = window.scrollY + window.innerHeight * 0.33;
  let activeId = "";
  for (const section of sections) {
    if (section.offsetTop <= marker) activeId = section.id;
  }
  navLinks.forEach((link) => {
    const isActive = link.getAttribute("href") === `#${activeId}`;
    link.classList.toggle("active", isActive);
    if (isActive) link.setAttribute("aria-current", "location");
    else link.removeAttribute("aria-current");
  });
}

let ticking = false;
window.addEventListener("scroll", () => {
  if (ticking) return;
  ticking = true;
  window.requestAnimationFrame(() => {
    updateHeader();
    updateActiveSection();
    ticking = false;
  });
}, { passive: true });

updateHeader();
updateActiveSection();
