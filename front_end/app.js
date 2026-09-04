// ---------------------------------------------------------------------------
// AMRGuard frontend — wired to the REAL backend (main.py)
// ---------------------------------------------------------------------------
const API_BASE = "http://localhost:8000/api";

const views = {
  landing:  document.getElementById("view-landing"),
  select:   document.getElementById("view-select"),
  pathogen: document.getElementById("view-pathogen"),
  empiric:  document.getElementById("view-empiric"),
  results:  document.getElementById("view-results"),
};

let lastView = "select";

function showView(name) {
  Object.values(views).forEach(v => { if (v) v.classList.remove("view-active"); });
  if (views[name]) views[name].classList.add("view-active");
  hideError();
  window.scrollTo({ top: 0, behavior: "smooth" });
}
function showError(message) {
  const banner = document.getElementById("error-banner");
  banner.textContent = message;
  banner.hidden = false;
}
function hideError() {
  document.getElementById("error-banner").hidden = true;
}

// ---------------------------------------------------------------------------
// Allergy / condition checkboxes — shared loader
// ---------------------------------------------------------------------------
let allergyOptionsCache = null;
let conditionOptionsCache = null;

async function loadCheckboxGroup(containerId, endpoint, cacheGetSet) {
  const container = document.getElementById(containerId);
  if (!container) return;

  let options = cacheGetSet.get();
  if (!options) {
    try {
      const res = await fetch(`${API_BASE}/${endpoint}`);
      if (!res.ok) throw new Error(`Could not load ${endpoint} list`);
      options = await res.json();
      cacheGetSet.set(options);
    } catch (err) {
      showError(err.message);
      return;
    }
  }

  container.innerHTML = "";
  options.forEach(a => {
    const label = document.createElement("label");
    label.className = "allergy-option";
    label.innerHTML = `<input type="checkbox" value="${a.id}"> ${a.name}`;
    container.appendChild(label);
  });
}

function loadAllergyCheckboxes(containerId) {
  return loadCheckboxGroup(containerId, "allergies", {
    get: () => allergyOptionsCache,
    set: (v) => { allergyOptionsCache = v; },
  });
}

function loadConditionCheckboxes(containerId) {
  return loadCheckboxGroup(containerId, "conditions", {
    get: () => conditionOptionsCache,
    set: (v) => { conditionOptionsCache = v; },
  });
}

function getSelectedValues(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return [];
  return Array.from(container.querySelectorAll("input[type=checkbox]:checked")).map(cb => cb.value);
}

// ---------------------------------------------------------------------------
// Path selection (step 0)
// ---------------------------------------------------------------------------
document.querySelectorAll(".path-card").forEach(card => {
  card.addEventListener("click", () => {
    const path = card.dataset.path;
    if (path === "known") {
      loadKnownForm();
      showView("pathogen");
    } else {
      loadEmpiricOptions();
      showView("empiric");
    }
  });
});

const enterBtn = document.getElementById("enter-tool");
if (enterBtn) {
  enterBtn.addEventListener("click", () => showView("select"));
}

document.querySelectorAll("[data-back]").forEach(btn => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.back.replace("view-", "");
    showView(target);
  });
});

// ---------------------------------------------------------------------------
// Path A: known pathogen  (country -> species)
// ---------------------------------------------------------------------------
async function loadKnownForm() {
  const countrySel = document.getElementById("known-country");
  const speciesSel = document.getElementById("known-species");
  speciesSel.innerHTML = '<option value="" disabled selected>Select country first…</option>';
  loadAllergyCheckboxes("known-allergies");
  loadConditionCheckboxes("known-conditions");

  try {
    const res = await fetch(`${API_BASE}/countries`);
    if (!res.ok) throw new Error("Could not load countries");
    const countries = await res.json();

    countrySel.querySelectorAll("option:not([disabled])").forEach(o => o.remove());
    countries.forEach(c => {
      const opt = document.createElement("option");
      opt.value = c.id;
      opt.textContent = c.name;
      countrySel.appendChild(opt);
    });
    countrySel.value = "";
  } catch (err) {
    showError(err.message);
  }
}

document.getElementById("known-country").addEventListener("change", async (e) => {
  const country = e.target.value;
  const speciesSel = document.getElementById("known-species");
  speciesSel.innerHTML = '<option value="" disabled selected>Loading…</option>';
  try {
    const res = await fetch(`${API_BASE}/species/${encodeURIComponent(country)}`);
    if (!res.ok) throw new Error("Could not load species");
    const species = await res.json();
    speciesSel.innerHTML = '<option value="" disabled selected>Select species…</option>';
    species.forEach(s => {
      const opt = document.createElement("option");
      opt.value = s;
      opt.textContent = s;
      speciesSel.appendChild(opt);
    });
  } catch (err) {
    showError(err.message);
  }
});

document.getElementById("known-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const country = document.getElementById("known-country").value;
  const species = document.getElementById("known-species").value;
  const allergies = getSelectedValues("known-allergies");
  const conditionsSelected = getSelectedValues("known-conditions");
  const excludeAllergens = document.getElementById("known-exclude-toggle").checked;
  const topN = document.getElementById("known-topn").value;

  let url = `${API_BASE}/recommend-known?country=${encodeURIComponent(country)}&species=${encodeURIComponent(species)}&top_n=${topN}`;
  if (allergies.length) url += `&allergies=${encodeURIComponent(allergies.join(","))}`;
  if (excludeAllergens) url += `&exclude_allergens=true`;
  if (conditionsSelected.length) url += `&conditions=${encodeURIComponent(conditionsSelected.join(","))}`;

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error("No recommendation found for that combination");
    const data = await res.json();
    lastView = "pathogen";
    document.getElementById("results-title").textContent = data.species;
    document.getElementById("results-sub").textContent =
      `${data.country} · based on ${data.year_used} surveillance data`;
    document.getElementById("ranking-note").hidden = true;   // known mode: no coverage ranking
    renderTable(data.recommendations, data.excluded_for_allergy);
    showView("results");
  } catch (err) {
    showError(err.message);
  }
});

// ---------------------------------------------------------------------------
// Path B: empiric  (country only)
// ---------------------------------------------------------------------------
async function loadEmpiricOptions() {
  const countrySelect = document.getElementById("country-select");
  loadAllergyCheckboxes("empiric-allergies");
  loadConditionCheckboxes("empiric-conditions");
  try {
    const res = await fetch(`${API_BASE}/countries`);
    if (!res.ok) throw new Error("Could not load countries");
    const countries = await res.json();
    countrySelect.querySelectorAll("option:not([disabled])").forEach(o => o.remove());
    countries.forEach(c => {
      const opt = document.createElement("option");
      opt.value = c.id;
      opt.textContent = c.name;
      countrySelect.appendChild(opt);
    });
    countrySelect.value = "";
  } catch (err) {
    showError(err.message);
  }
}

document.getElementById("empiric-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const country = document.getElementById("country-select").value;
  const allergies = getSelectedValues("empiric-allergies");
  const conditionsSelected = getSelectedValues("empiric-conditions");
  const excludeAllergens = document.getElementById("empiric-exclude-toggle").checked;
  const topN = document.getElementById("empiric-topn").value;

  let url = `${API_BASE}/recommend-empiric?country=${encodeURIComponent(country)}&top_n=${topN}`;
  if (allergies.length) url += `&allergies=${encodeURIComponent(allergies.join(","))}`;
  if (excludeAllergens) url += `&exclude_allergens=true`;
  if (conditionsSelected.length) url += `&conditions=${encodeURIComponent(conditionsSelected.join(","))}`;

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error("No empiric data found for that country");
    const data = await res.json();
    lastView = "empiric";
    document.getElementById("results-title").textContent = `Empiric guidance — ${data.country}`;
    document.getElementById("results-sub").textContent =
      `Ranked across circulating pathogens · based on ${data.year_used} surveillance data`;
    document.getElementById("ranking-note").hidden = false;  // empiric mode: show coverage explanation
    renderTable(data.recommendations, data.excluded_for_allergy);
    showView("results");
  } catch (err) {
    showError(err.message);
  }
});

// ---------------------------------------------------------------------------
// Shared: render results table
// ---------------------------------------------------------------------------
function renderTable(rows, excludedCount) {
  const tbody = document.getElementById("rx-table-body");
  tbody.innerHTML = "";

  const confClass = { High: "susc-S", Moderate: "susc-I", Low: "susc-R", "Very low": "susc-R", Unknown: "" };
  const awareClass = { Access: "susc-S", Watch: "susc-I", Reserve: "susc-R" };
  const levelIcon = { routine: "•", moderate: "⚠", serious: "⚠" };

  rows.forEach(row => {
    const cov = row.coverage_pct != null ? ` · covers ~${row.coverage_pct}%` : "";
    const nIso = row.n_isolates != null ? `${row.n_isolates} isolates` : "—";
    const awareBadge = row.aware
      ? `<span class="susc-badge ${awareClass[row.aware] || ""}">${row.aware}</span>`
      : "—";
    const discontinuedBadge = row.discontinued
      ? `<div class="caution caution-serious">⚠ ${row.discontinued}</div>`
      : "";
    const allergyBadge = row.allergy_flag
      ? `<div class="caution caution-moderate">⚠ Possible allergen — verify specific allergy</div>`
      : "";
    const conditionBadges = (row.condition_cautions || [])
      .map(c => `<div class="caution caution-${c.level}">${levelIcon[c.level] || "•"} ${c.reason}</div>`)
      .join("");
    const tr = document.createElement("tr");
    if (row.allergy_flag || (row.condition_cautions || []).length || row.discontinued) {
      tr.classList.add("row-allergy-flag");
    }
    tr.innerHTML = `
      <td class="drug-name">${row.drug}${discontinuedBadge}${allergyBadge}${conditionBadges}</td>
      <td><strong>${row.susceptibility_pct}%</strong></td>
      <td><span class="susc-badge ${confClass[row.confidence] || ""}">${row.confidence}</span>
          <div class="note-cell">${nIso}${cov}</div></td>
      <td>${awareBadge}</td>
    `;
    tbody.appendChild(tr);
  });

  const noteEl = document.getElementById("allergy-excluded-note");
  if (noteEl) {
    if (excludedCount) {
      noteEl.textContent = `${excludedCount} drug${excludedCount === 1 ? "" : "s"} hidden due to stated allergies.`;
      noteEl.hidden = false;
    } else {
      noteEl.hidden = true;
    }
  }
}

document.getElementById("results-back").addEventListener("click", () => {
  showView(lastView);
});

// ---- rotating antibiotic molecule on the landing page ----
function initMoleculeBg() {
  const el = document.getElementById("mol-bg");
  if (!el || typeof $3Dmol === "undefined") return;
  const viewer = $3Dmol.createViewer(el, { backgroundColor: 0x000000, backgroundAlpha: 0 });
  fetch("CDK1.pdb")
    .then(r => r.text())
    .then(data => {
      viewer.addModel(data, "pdb");
      viewer.setStyle({}, { cartoon: { color: "spectrum" } });
      viewer.zoomTo();
      viewer.spin("y", 1);
      viewer.render();
    })
    .catch((e) => console.log("molecule load failed:", e));
}
initMoleculeBg();
