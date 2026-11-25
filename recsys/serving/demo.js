"use strict";

const state = {
  profile: null, recommendations: [], rows: [], currentView: "home",
  currentType: "all", search: "", hero: null, artwork: false,
};
const $ = selector => document.querySelector(selector);
const esc = value => String(value ?? "").replace(/[&<>"']/g, char => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[char]));
let catalogController, searchTimer, detailVersion = 0, lastFocus;

function status(message = "") {
  $("#status").textContent = message;
  $("#status").hidden = !message;
}

async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    if (response.status === 503) throw new Error("Recommendations are not ready. Build the demo model and restart the server.");
    if (response.status === 404) throw new Error("This title is no longer available.");
    throw new Error("Could not load this content. Please try again.");
  }
  return response.json();
}

function meta(item) {
  return [item.release_year, item.maturity_rating,
    item.runtime || (item.seasons ? `${item.seasons} season${item.seasons === 1 ? "" : "s"}` : "")
  ].filter(Boolean).join(" • ");
}

function sourceLabel(item) {
  return ({collaborative: "Learned pick", content: "Genre discovery",
    popularity: "Popular pick", model_bias: "Starter pick"})[item.source] || "Explore";
}

function card(item) {
  const label = Number.isFinite(item.rating) ? `Rated ${item.rating}/5` : sourceLabel(item);
  const artwork = state.artwork && item.poster_url
    ? `<img class="poster" loading="lazy" src="${esc(item.poster_url)}" alt="${esc(item.title)} poster" />`
    : `<div class="poster-fallback"><span>${esc(item.title)}</span></div>`;
  return `<button class="title-card" data-title-id="${esc(item.item_id)}" aria-label="View details for ${esc(item.title)}"><div class="poster-wrap">${artwork}<span class="badge">${item.type === "series" ? "SERIES" : "FILM"}</span><span class="match">${esc(label)}</span></div><div class="card-title">${esc(item.title)}</div><p class="card-meta">${esc(meta(item))}</p></button>`;
}

function rail(title, caption, items) {
  return `<section class="section"><div class="section-head"><h2>${esc(title)}</h2><span class="section-caption">${esc(caption)}</span></div><div class="rail">${items.map(card).join("")}</div></section>`;
}

function setArtwork(selector, item) {
  const url = state.artwork && (item?.backdrop_url || item?.poster_url);
  $(selector).style.backgroundImage = url ? `url("${url}")` : "none";
}

function renderProfile() {
  const genres = state.profile.top_genres || [];
  const name = $("#user").selectedOptions[0]?.textContent || "Viewer";
  $("#profileHeading").textContent = `${name} is into ${genres.slice(0, 2).map(item => item.genre).join(" and ") || "great stories"}`;
  $("#tastePills").innerHTML = genres.map(item => `<span class="pill">${esc(item.genre)}</span>`).join("") || '<span class="pill">Building your taste</span>';
  const history = state.profile.history || [];
  $("#profileReason").textContent = history.length
    ? `Positive-rating history includes titles such as ${history[0].title}.`
    : "No positive-rating history yet. Start with popular picks and catalogue discoveries.";
}

function renderHero() {
  const item = state.hero;
  $("#heroDetails").disabled = !item;
  $("#heroTitle").textContent = item?.title || "No picks available";
  $("#heroKicker").textContent = item ? sourceLabel(item) : "Try another profile";
  $("#heroMeta").textContent = item ? [meta(item), ...(item.genres || []).slice(0, 3)].filter(Boolean).join(" • ") : "";
  $("#heroDescription").textContent = item?.reason || "There are no eligible recommendations for this profile.";
  setArtwork("#heroArt", item);
}

function renderHome() {
  $("#homeView").innerHTML = rail("Picked for your profile", "Learned preferences + genre discoveries", state.recommendations)
    + state.rows.map(row => rail(row.title, "", row.items)).join("");
}

function renderTaste() {
  const profile = state.profile;
  if (!profile) return;
  const history = profile.history || [], genres = profile.top_genres || [];
  const sample = profile.genre_sample_size || 0;
  const heading = profile.has_learned_profile ? "Your learned taste profile" : "Your taste starts here";
  const stats = [[profile.observed_count, "Ratings in this demo history"],
    [profile.positive_count, `Rated ${profile.rating_threshold}+ stars`],
    [profile.has_learned_profile ? "Learned" : "Starter", "Recommendation mode"]];
  const bars = genres.map(row => `<div class="genre-row"><span>${esc(row.genre)}</span><div class="genre-track"><span class="genre-fill" style="width:${Math.round(100 * row.count / Math.max(1, sample))}%"></span></div><span>${row.count}/${sample}</span></div>`).join("");
  $("#tasteView").innerHTML = `<div class="section-head"><h2>${heading}</h2></div><p class="taste-copy">${profile.has_learned_profile ? "The recommendation model learns from positive ratings in this profile. Genre summaries help explain the additional ranking signals." : "This viewer has no learned rating profile yet. Popular picks offer a starting point."}</p><div class="taste-stats">${stats.map(([value, label]) => `<div class="taste-stat"><strong>${esc(value)}</strong><span>${esc(label)}</span></div>`).join("")}</div>${sample ? `<div class="section-head"><h2>Genres in your positive ratings</h2></div><p class="taste-copy">Counts from the ${sample} highest-rated titles, with recent ratings first when tied. A title can belong to several genres. These counts describe history, not the model's internal feature weights.</p><div class="genre-bars">${bars}</div>` : ""}${history.length ? rail("Movies you rated highly", `Showing ${history.length} of ${profile.positive_count} positive ratings`, history) : '<p class="taste-copy">No positively rated movies to show for this viewer.</p>'}${rail(profile.has_learned_profile ? "Recommended for this profile" : "Popular picks to explore", "Previously rated titles excluded", state.recommendations)}`;
}

async function loadRecommendations() {
  $("#refresh").disabled = true;
  $("#user").disabled = true;
  status("Updating recommendations…");
  const user = $("#user").value;
  try {
    const [profile, data] = await Promise.all([
      request(`/profile/${encodeURIComponent(user)}`),
      request("/recommend", {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({user_id: user, k: 18, context: {surface: "reelfind_home"}})}),
    ]);
    state.profile = profile;
    state.recommendations = data.items;
    state.hero = data.items[0] || null;
    renderProfile(); renderHero(); renderHome(); renderTaste();
    status();
  } catch (error) {
    // Clear previous-user content rather than showing it under a new profile name.
    state.profile = null; state.recommendations = []; state.hero = null;
    $("#profileHeading").textContent = "Profile could not be loaded";
    $("#profileReason").textContent = "Retry with Refresh picks.";
    $("#tastePills").innerHTML = "";
    $("#tasteView").innerHTML = '<p class="taste-copy">Profile unavailable. Please retry.</p>';
    renderHero(); renderHome(); status(error.message);
  } finally {
    $("#refresh").disabled = false; $("#user").disabled = false;
  }
}

function renderFilters() {
  $("#filters").innerHTML = [["all", "All titles"], ["movie", "Films"], ["series", "Series"]]
    .map(([value, label]) => `<button class="filter ${state.currentType === value ? "active" : ""}" data-filter="${value}">${label}</button>`).join("");
}

async function loadCatalog() {
  catalogController?.abort();
  const controller = new AbortController();
  catalogController = controller;
  try {
    const params = new URLSearchParams({media_type: state.currentType, query: state.search, limit: 80});
    const data = await request(`/catalog?${params}`, {signal: controller.signal});
    if (controller.signal.aborted) return;
    $("#catalog").innerHTML = data.items.length ? data.items.map(card).join("") : '<div class="empty">No titles found. Try a different search.</div>';
    $("#catalogCount").textContent = `${data.items.length} titles`;
    status();
  } catch (error) {
    if (error.name !== "AbortError") status(error.message);
  }
}

async function showView(view) {
  state.currentView = view;
  catalogController?.abort();
  document.querySelector("main").classList.toggle("taste-mode", view !== "home");
  document.querySelectorAll(".nav button").forEach(button => button.classList.toggle("active", button.dataset.view === view));
  $("#homeView").classList.toggle("hidden", view !== "home");
  $("#tasteView").classList.toggle("hidden", view !== "taste");
  $("#catalogView").classList.toggle("hidden", !["movies", "series", "search"].includes(view));
  $("#profileNote").classList.toggle("hidden", !["home", "taste"].includes(view));
  $("#hero").classList.toggle("hidden", view !== "home");
  if (view === "home") return;
  if (view === "taste") { renderTaste(); return; }
  state.currentType = view === "movies" ? "movie" : view === "series" ? "series" : "all";
  $("#catalogTitle").textContent = view === "search" ? "Search results" : view === "movies" ? "Films" : "Series";
  renderFilters(); await loadCatalog();
}

async function showDetails(itemId) {
  const version = ++detailVersion;
  const data = await request(`/title/${encodeURIComponent(itemId)}`);
  if (version !== detailVersion) return;
  const item = data.item;
  $("#detailTitle").textContent = item.title;
  $("#detailType").textContent = `${item.type === "series" ? "Series" : "Film"}${data.tagline ? ` • ${data.tagline}` : ""}`;
  $("#detailMeta").textContent = [meta(item), ...(item.genres || [])].filter(Boolean).join(" • ");
  $("#detailOverview").textContent = data.overview || item.overview || "";
  const pick = state.recommendations.find(row => row.item_id === itemId);
  $("#detailReason").textContent = pick ? `Why this pick: ${pick.reason}` : "";
  $("#detailReason").classList.toggle("hidden", !pick);
  $("#detailCredits").textContent = data.credits || "Not in the local catalogue";
  $("#detailCreator").textContent = data.creator || "Not in the local catalogue";
  $("#detailGenres").textContent = (item.genres || []).join(", ");
  setArtwork("#detailArt", item);
  $("#related").innerHTML = (data.related || []).map(card).join("");
  if (!$("#detail").classList.contains("open")) lastFocus = document.activeElement;
  $("#detail").classList.add("open");
  $("#detail").scrollTop = 0;
  document.body.style.overflow = "hidden";
  document.querySelector("main").inert = true;
  document.querySelector("header").inert = true;
  document.querySelector("footer").inert = true;
  $("#detailClose").focus();
}

function closeDetails() {
  ++detailVersion;
  $("#detail").classList.remove("open");
  document.body.style.overflow = "";
  document.querySelectorAll("[inert]").forEach(element => { element.inert = false; });
  if (lastFocus?.isConnected) lastFocus.focus();
}

document.addEventListener("click", async event => {
  try {
    const cardEl = event.target.closest("[data-title-id]");
    if (cardEl) return await showDetails(cardEl.dataset.titleId);
    const nav = event.target.closest("[data-view]");
    if (nav) {
      clearTimeout(searchTimer); state.search = ""; $("#search").value = "";
      return await showView(nav.dataset.view);
    }
    const filter = event.target.closest("[data-filter]");
    if (filter) { state.currentType = filter.dataset.filter; renderFilters(); await loadCatalog(); }
  } catch (error) { status(error.message); }
});

$("#refresh").addEventListener("click", loadRecommendations);
$("#user").addEventListener("change", loadRecommendations);
$("#heroDetails").addEventListener("click", () => {
  if (state.hero) showDetails(state.hero.item_id).catch(error => status(error.message));
});
$("#detailClose").addEventListener("click", closeDetails);
$("#detailBack").addEventListener("click", closeDetails);
$("#homeBrand").addEventListener("click", () => {
  state.search = ""; $("#search").value = ""; clearTimeout(searchTimer); showView("home");
});
$("#search").maxLength = 200;
$("#search").addEventListener("input", event => {
  clearTimeout(searchTimer); catalogController?.abort();
  state.search = event.target.value.trim();
  searchTimer = setTimeout(() => showView(state.search ? "search" : "home"), 200);
});

document.addEventListener("keydown", event => {
  if (!$("#detail").classList.contains("open")) return;
  if (event.key === "Escape") { event.preventDefault(); closeDetails(); }
  if (event.key === "Tab") {
    const focusable = [...$("#detail").querySelectorAll("button, a[href], input, select")].filter(element => !element.disabled);
    const first = focusable[0], last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  }
});

document.addEventListener("error", event => {
  const image = event.target;
  if (!(image instanceof HTMLImageElement) || !image.classList.contains("poster")) return;
  const fallback = document.createElement("div");
  fallback.className = "poster-fallback";
  const text = document.createElement("span");
  text.textContent = image.alt.replace(/ poster$/, "");
  fallback.append(text); image.replaceWith(fallback);
}, true);

(async function boot() {
  try {
    const [users, model, browse] = await Promise.all([
      request("/sample-users?limit=40"), request("/model-info"), request("/browse-rows?limit=18"),
    ]);
    state.rows = browse.rows || []; state.artwork = Boolean(model.remote_artwork);
    $("#user").innerHTML = users.users.map((id, index) => `<option value="${esc(id)}">Profile ${index + 1}</option>`).join("") + '<option value="demo-new-viewer">New viewer</option>';
    $("#modelNote").textContent = model.ready
      ? `${model.validation_selected ? "Validation-selected" : "Baseline"} BPR model · ${model.factors} learned features per user and movie · positive ratings: ${model.rating_threshold}+ stars.`
      : "Build a model to enable learned recommendations.";
    await loadRecommendations();
  } catch (error) { $("#heroTitle").textContent = "The catalogue needs a moment"; status(error.message); }
})();
