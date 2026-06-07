document.addEventListener("DOMContentLoaded", function () {
  // 1. Target the top-left logo area in PyData Sphinx Theme
  const navbarStart = document.querySelector(".navbar-header-items__start");
  if (!navbarStart) return;

  // 2. Extract current version and determine path context from the URL
  const pathSegments = window.location.pathname.split("/").filter(Boolean);
  const repoIndex = pathSegments.indexOf("swectral");
  
  const versionIndex = repoIndex !== -1 ? repoIndex + 1 : 0;
  const currentVersion = pathSegments[versionIndex] ? pathSegments[versionIndex] : "v0.6.5";
  
  const basePath = repoIndex !== -1 
    ? `/swectral/${currentVersion}/` 
    : `/${currentVersion}/`;

  // 3. Fetch the manifest using an explicit version root path
  fetch(`${basePath}_static/switcher.json`)
    .then(response => {
      if (!response.ok) throw new Error("Manifest not found at base target");
      return response;
    })
    .then(response => response.json())
    .then(data => {
      // Extract the release versions, but STRICTLY filter out 'master' 
      // This hides the development branch from the public dropdown completely
      const versions = data
        .map(item => item.version)
        .filter(v => v !== "master");

      // 4. Construct the Bootstrap dropdown container element
      const container = document.createElement("div");
      container.className = "version-switcher__container dropdown ms-3 align-self-center";

      const rootUrlPrefix = repoIndex !== -1 ? "/swectral" : "";

      let itemsHtml = versions.map(v => `
        <a class="dropdown-item list-group-item list-group-item-action py-1 ${v === currentVersion ? 'active' : ''}" 
           href="${rootUrlPrefix}/${v}/index.html">
          ${v}
        </a>
      `).join("");

      // Mark the top button text as development if viewing the master branch backdoor link
      const displayVersion = currentVersion === "master" ? "development (master)" : currentVersion;

      container.innerHTML = `
        <button id="global-version-switcher" type="button" 
                class="btn btn-sm dropdown-toggle" data-bs-toggle="dropdown" 
                aria-haspopup="true" aria-expanded="false" style="border: 1px solid var(--pst-color-border, #ccc);">
          ${displayVersion}
        </button>
        <div class="dropdown-menu list-group-flush py-0" aria-labelledby="global-version-switcher">
          ${itemsHtml}
        </div>
      `;

      // Inject the fully assembled button container directly next to your logo title
      navbarStart.appendChild(container);

      // 5. NEW: Inject a global prominent warning banner if on the master (dev) branch
      if (currentVersion === "master") {
        const latestStable = versions.length > 0 ? versions[0] : "v0.6.5";
        
        const warningBanner = document.createElement("div");
        // Uses native PyData theme alert layout styles to match perfectly
        warningBanner.className = "bd-header-announcement container-fluid bg-warning text-dark text-center py-2 font-weight-bold";
        warningBanner.style.fontSize = "0.9rem";
        warningBanner.style.borderBottom = "1px solid rgba(0,0,0,0.1)";
        warningBanner.innerHTML = `
          ⚠️ <strong>Notice:</strong> You are viewing the documentation for an unreleased <strong>development version</strong>. 
          Features here may be unstable or preliminary. 
          For stable production, switch to the <a href="${rootUrlPrefix}/${latestStable}/index.html" style="text-decoration: underline; color: #004085; font-weight: 700;">latest release (${latestStable})</a>.
        `;

        // Places the notice bar at the very top of the page viewport, right above the navigation headers
        document.body.insertBefore(warningBanner, document.body.firstChild);
      }
    })
    .catch(err => console.error("Error loading version switcher:", err));
});