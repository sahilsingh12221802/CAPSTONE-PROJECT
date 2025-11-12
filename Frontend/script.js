const designTokens = {
  primaryColor: "#14B8A6", // teal-500
  secondaryColor: "#1E40AF", // blue-800
  accentColor: "#FACC15", // yellow-400
  background: "#F9FAFB", // gray-50
  text: "#111827", // gray-900
  border: "#E5E7EB", // gray-200
};



const elements = {
  brandTitle: document.getElementById("brand-title"),
  brandSubtitle: document.getElementById("brand-subtitle"),
  footerText: document.getElementById("footer-text"),
  pageTitle: document.getElementById("page-title"),
  sidebar: document.getElementById("sidebar"),
  mobileMenu: document.getElementById("mobile-menu"),
  navLinks: document.querySelectorAll(".nav-link"),
  content: document.getElementById("content"),
  sections: document.querySelectorAll("main section"),
};



async function initSDK() {
  if (!window.ElementSDK || !window.DataSDK) {
    console.warn("SDKs not loaded — using local fallback.");
    return;
  }

  try {
    const element = await window.ElementSDK.init();
    const data = await window.DataSDK.init();

    console.log("SDK Initialized:", { element, data });
  } catch (err) {
    console.error("SDK initialization failed:", err);
  }
}

initSDK();



function applyTheme(tokens) {
  document.documentElement.style.setProperty("--primary-color", tokens.primaryColor);
  document.documentElement.style.setProperty("--secondary-color", tokens.secondaryColor);
  document.documentElement.style.setProperty("--accent-color", tokens.accentColor);
  document.documentElement.style.setProperty("--background-color", tokens.background);
  document.documentElement.style.setProperty("--text-color", tokens.text);
  document.documentElement.style.setProperty("--border-color", tokens.border);
}

/* Apply theme immediately */
applyTheme(designTokens);



function setBranding() {
  elements.brandTitle.textContent = "AnimalVision AI";
  elements.brandSubtitle.textContent = "Image-Based Animal Classification";
  elements.footerText.textContent = "© 2025 AnimalVision. All rights reserved.";
}

setBranding();


function navigateTo(route) {
  elements.sections.forEach((section) => {
    section.classList.toggle("hidden", section.id !== route);
  });

  elements.pageTitle.textContent = route.charAt(0).toUpperCase() + route.slice(1);
  localStorage.setItem("activeRoute", route);
}


const savedRoute = localStorage.getItem("activeRoute") || "home";
navigateTo(savedRoute);


elements.navLinks.forEach((link) => {
  link.addEventListener("click", () => {
    const route = link.getAttribute("data-route");
    navigateTo(route);
    updateActiveNav(link);
  });
});

function updateActiveNav(activeLink) {
  elements.navLinks.forEach((link) => {
    link.classList.toggle("bg-gray-100", link === activeLink);
    link.classList.toggle("text-teal-600", link === activeLink);
    link.classList.toggle("font-semibold", link === activeLink);
  });
}


elements.mobileMenu.addEventListener("click", () => {
  elements.sidebar.classList.toggle("hidden");
  elements.sidebar.classList.toggle("absolute");
  elements.sidebar.classList.toggle("z-50");
  elements.sidebar.classList.toggle("bg-white");
  elements.sidebar.classList.toggle("shadow-xl");
});


const searchInput = document.getElementById("search_input");

if (searchInput) {
  searchInput.addEventListener("input", (e) => {
    const query = e.target.value.toLowerCase();
    console.log("Searching for:", query);

    // Optional: implement filtering within results section
    if (query.length > 2) {
      const results = document.querySelectorAll("#results ul li");
      results.forEach((item) => {
        const text = item.textContent.toLowerCase();
        item.classList.toggle("hidden", !text.includes(query));
      });
    }
  });
}


const fileInput = document.getElementById("fileInput");
if (fileInput) {
  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (evt) => {
      console.log("File uploaded:", file.name);
      // TODO: send to model for classification
    };
    reader.readAsDataURL(file);
  });
}
