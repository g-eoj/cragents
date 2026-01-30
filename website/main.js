// Add copy buttons to code blocks
function addCopyButtons() {
  document.querySelectorAll('pre').forEach(pre => {
    const wrapper = document.createElement('div');
    wrapper.className = 'code-block';
    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.appendChild(pre);

    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'Copy';
    btn.addEventListener('click', async () => {
      const code = pre.querySelector('code')?.textContent || pre.textContent;
      await navigator.clipboard.writeText(code);
      btn.textContent = 'Copied!';
      setTimeout(() => btn.textContent = 'Copy', 2000);
    });
    wrapper.appendChild(btn);
  });
}

// Split content into sections with scroll-based navigation
document.addEventListener('DOMContentLoaded', () => {
  const main = document.querySelector('main');
  const content = main.innerHTML;

  // Parse content and split by h2 headers
  const temp = document.createElement('div');
  temp.innerHTML = content;

  const sections = [];
  let currentSection = null;

  const h1 = temp.querySelector('h1');
  const introContent = [];

  for (const child of Array.from(temp.children)) {
    if (child.tagName === 'H1') {
      continue;
    }
    if (child.tagName === 'H2') {
      if (currentSection) {
        sections.push(currentSection);
      }
      const id = child.textContent.toLowerCase().replace(/[^a-z0-9]+/g, '-');
      currentSection = {
        id,
        title: child.textContent,
        content: [child.outerHTML]
      };
    } else if (currentSection) {
      currentSection.content.push(child.outerHTML);
    } else {
      introContent.push(child.outerHTML);
    }
  }

  if (currentSection) {
    sections.push(currentSection);
  }

  if (introContent.length > 0 || h1) {
    sections.unshift({
      id: 'intro',
      title: 'Intro',
      content: [h1 ? h1.outerHTML : '', ...introContent]
    });
  }

  // Get title from h1
  const title = h1 ? h1.textContent.trim() : 'cragents';

  // Create sidebar
  const sidebar = document.createElement('aside');
  sidebar.className = 'sidebar';
  sidebar.innerHTML = `
    <div class="sidebar-title">${title}</div>
    <nav class="sidebar-nav">
      ${sections.map((s, i) => `
        <button class="nav-btn${i === 0 ? ' active' : ''}" data-section="${s.id}" data-index="${i}">
          ${s.title}
        </button>
      `).join('')}
      <a href="https://github.com/g-eoj/cragents" class="nav-btn github-link" aria-label="GitHub">
        <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
          <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
        </svg>
      </a>
    </nav>
  `;

  // Create section containers
  const sectionsHtml = sections.map((s, i) => `
    <section id="section-${s.id}" class="content-section${i === 0 ? ' active' : ''}" data-index="${i}">
      <div class="section-inner">
        ${s.content.join('\n')}
      </div>
    </section>
  `).join('\n');

  // Update DOM
  main.innerHTML = sectionsHtml;
  document.body.insertBefore(sidebar, main);
  addCopyButtons();

  // State
  let currentIndex = 0;
  let isTransitioning = false;
  const buttons = sidebar.querySelectorAll('.nav-btn');
  const sectionEls = main.querySelectorAll('.content-section');

  // Check if sections are too tall and mark them
  function updateTallSections() {
    const viewportHeight = main.clientHeight;
    sectionEls.forEach(sec => {
      const inner = sec.querySelector('.section-inner');
      if (inner.scrollHeight > viewportHeight) {
        sec.classList.add('tall');
      } else {
        sec.classList.remove('tall');
      }
    });
  }

  updateTallSections();
  window.addEventListener('resize', updateTallSections);

  // Set initial position classes
  sectionEls.forEach((sec, i) => {
    if (i < currentIndex) {
      sec.classList.add('above');
    }
  });

  function showSection(index) {
    if (index < 0 || index >= sections.length || isTransitioning || index === currentIndex) return;

    isTransitioning = true;
    currentIndex = index;

    // Update nav
    buttons.forEach((btn, i) => {
      btn.classList.toggle('active', i === index);
    });

    // Update section classes based on position relative to current
    sectionEls.forEach((sec, i) => {
      if (i === index) {
        sec.classList.add('active');
        sec.classList.remove('above');
        sec.scrollTop = 0; // Reset scroll position
      } else if (i < index) {
        sec.classList.remove('active');
        sec.classList.add('above');
      } else {
        sec.classList.remove('active', 'above');
      }
    });

    // Update URL
    history.replaceState(null, '', `#${sections[index].id}`);

    // Wait for transition
    setTimeout(() => {
      isTransitioning = false;
    }, 600);
  }

  // Nav button clicks
  buttons.forEach((btn, i) => {
    btn.addEventListener('click', () => {
      showSection(i);
    });
  });

  // Scroll/wheel handling
  let scrollAccumulator = 0;
  const scrollThreshold = 50;

  function handleWheel(e) {
    if (isTransitioning) {
      e.preventDefault();
      return;
    }

    const activeSection = sectionEls[currentIndex];
    const isTall = activeSection.classList.contains('tall');

    if (isTall) {
      const atTop = activeSection.scrollTop <= 0;
      const atBottom = activeSection.scrollTop + activeSection.clientHeight >= activeSection.scrollHeight - 1;

      // Allow native scroll within tall sections
      if (e.deltaY > 0 && !atBottom) {
        return; // Scrolling down, not at bottom - allow native scroll
      }
      if (e.deltaY < 0 && !atTop) {
        return; // Scrolling up, not at top - allow native scroll
      }
    }

    e.preventDefault();
    scrollAccumulator += e.deltaY;

    if (Math.abs(scrollAccumulator) >= scrollThreshold) {
      if (scrollAccumulator > 0 && currentIndex < sections.length - 1) {
        showSection(currentIndex + 1);
      } else if (scrollAccumulator < 0 && currentIndex > 0) {
        showSection(currentIndex - 1);
      }
      scrollAccumulator = 0;
    }
  }

  document.addEventListener('wheel', handleWheel, { passive: false });

  // Touch handling for mobile
  let touchStartY = 0;

  document.addEventListener('touchstart', (e) => {
    touchStartY = e.touches[0].clientY;
  }, { passive: true });

  document.addEventListener('touchend', (e) => {
    if (isTransitioning) return;

    const activeSection = sectionEls[currentIndex];
    const isTall = activeSection.classList.contains('tall');

    if (isTall) {
      const atTop = activeSection.scrollTop <= 0;
      const atBottom = activeSection.scrollTop + activeSection.clientHeight >= activeSection.scrollHeight - 1;

      const touchEndY = e.changedTouches[0].clientY;
      const diff = touchStartY - touchEndY;

      if (diff > 0 && !atBottom) return;
      if (diff < 0 && !atTop) return;
    }

    const touchEndY = e.changedTouches[0].clientY;
    const diff = touchStartY - touchEndY;

    if (Math.abs(diff) > 50) {
      if (diff > 0 && currentIndex < sections.length - 1) {
        showSection(currentIndex + 1);
      } else if (diff < 0 && currentIndex > 0) {
        showSection(currentIndex - 1);
      }
    }
  }, { passive: true });

  // Keyboard navigation
  document.addEventListener('keydown', (e) => {
    if (isTransitioning) return;

    if (e.key === 'ArrowDown' || e.key === 'PageDown') {
      e.preventDefault();
      if (currentIndex < sections.length - 1) {
        showSection(currentIndex + 1);
      }
    } else if (e.key === 'ArrowUp' || e.key === 'PageUp') {
      e.preventDefault();
      if (currentIndex > 0) {
        showSection(currentIndex - 1);
      }
    }
  });

  // Handle initial hash
  if (window.location.hash) {
    const id = window.location.hash.slice(1);
    const index = sections.findIndex(s => s.id === id);
    if (index > 0) {
      currentIndex = index;
      sectionEls.forEach((sec, i) => {
        sec.classList.toggle('active', i === index);
        sec.classList.toggle('above', i < index);
      });
      buttons.forEach((btn, i) => btn.classList.toggle('active', i === index));
    }
  }
});
