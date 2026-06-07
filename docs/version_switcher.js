document.addEventListener("DOMContentLoaded", function () {
  console.log("[Version Switcher] Script initialized.");

  // Prevent duplicate injections (from your brilliant addition!)
  if (document.getElementById("global-version-switcher")) {
    console.log("[Version Switcher] Already injected, skipping.");
    return;
  }

  // 1. Target inside the navbar brand container so it sits right next to the title text
  const logoItem = document.querySelector(".navbar-header-items__start .navbar-item") || 
                   document.querySelector(".navbar-brand") ||
                   document.querySelector(".navbar-header-items__start");
  
  if (!logoItem) {
    console.error("[Version Switcher] Failed to find logo navbar item in theme layout.");
    return;
  }

  // 2. Extract current version and determine path context from the URL
  const pathSegments = window.location.pathname.split("/").filter(Boolean);
  const repoIndex = pathSegments.indexOf("swectral");
  
  const versionIndex = repoIndex !== -1 ? repoIndex + 1 : 0;
  const currentVersion = pathSegments[versionIndex] ? pathSegments[versionIndex] : "v0.6.5";
  
  const basePath = repoIndex !== -1 
    ? `/swectral/${currentVersion}/` 
    : `/${currentVersion}/`;

  const targetJson = `${basePath}_static/switcher.json`;
  console.log(`[Version Switcher] Requesting manifest from: ${targetJson}`);

  // 3. Fetch the manifest
  fetch(targetJson)
    .then(response => {
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status} at ${targetJson}`);
      return response;
    })
    .then(response => response.json())
    .then(data => {
      // Extract the release versions, strictly filtering out 'master' 
      const versions = data
        .map(item => item.version)
        .filter(v => v !== "master");

      // 4. Construct the Bootstrap dropdown container element
      const container = document.createElement("div");
      // Use 'd-inline-block' to gracefully stand side-by-side with the logo title text
      container.className = "version-switcher__container dropdown d-inline-block ms-3 align-self-center";

      const rootUrlPrefix = repoIndex !== -1 ? "/swectral" : "";

      let itemsHtml = versions.map(v => `
        <a class="dropdown-item list-group-item list-group-item-action py-1 ${v === currentVersion ? 'active' : ''}" 
           href="${rootUrlPrefix}/${v}/index.html">
          ${v}
        </a>
      `).join("");

      const displayVersion = currentVersion === "master" ? "development (master)" : currentVersion;

      container.innerHTML = `
        <button id="global-version-switcher" type="button" 
                class="btn btn-sm dropdown-toggle" data-bs-toggle="dropdown" 
                aria-haspopup="true" aria-expanded="false" style="border: 1px solid var(--pst-color-border, #ccc); cursor: pointer; vertical-align: middle;">
          ${displayVersion}
        </button>
        <div class="dropdown-menu list-group-flush py-0" aria-labelledby="global-version-switcher">
          ${itemsHtml}
        </div>
      `;

      // 5. Inject the container directly into the logo wrapper element
      logoItem.appendChild(container);
      console.log("[Version Switcher] Dropdown successfully injected next to title.");

      // 6. Inject global warning banner if on master branch
      if (currentVersion === "master") {
        const latestStable = versions.length > 0 ? versions[0] : "v0.6.5";
        
        const warningBanner = document.createElement("div");
        warningBanner.className = "bd-header-announcement container-fluid bg-warning text-dark text-center py-2 font-weight-bold";
        warningBanner.style.fontSize = "0.9rem";
        warningBanner.style.borderBottom = "1px solid rgba(0,0,0,0.1)";
        warningBanner.style.position = "relative";
        warningBanner.style.zIndex = "1050";
        warningBanner.innerHTML = `
          ⚠️ <strong>Notice:</strong> You are viewing the documentation for an unreleased <strong>development version</strong>. 
          Features here may be unstable or preliminary. 
          For stable production, switch to the <a href="${rootUrlPrefix}/${latestStable}/index.html" style="text-decoration: underline; color: #004085; font-weight: 700;">latest release (${latestStable})</a>.
        `;

        document.body.insertBefore(warningBanner, document.body.firstChild);
        console.log("[Version Switcher] Master warning banner injected.");
      }
    })
    .catch(err => console.error("[Version Switcher] Critical error execution stopped:", err));
});