/* F1 Analytics Hub — motion and page interactions.
   Everything degrades to a plain, working page if JavaScript is unavailable. */

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

document.addEventListener("DOMContentLoaded", () => {
    initParallax();
    initReveal();
    initCounters();
    initBars();
    initCountdown();
    initNavToggle();
    initRaceSelectors();
    initFormFeedback();
});

/* --------------------------------------------------------------------------
   Parallax: scroll offset and pointer position are published as CSS custom
   properties, so the layers themselves are positioned entirely in CSS.
   -------------------------------------------------------------------------- */

function initParallax() {
    const root = document.documentElement;
    let scrollQueued = false;

    const applyScroll = () => {
        const y = window.scrollY;
        const max = Math.max(document.body.scrollHeight - window.innerHeight, 1);
        root.style.setProperty("--scroll-y", String(y));
        root.style.setProperty("--scroll-progress", String(Math.min(y / max, 1)));
        scrollQueued = false;
    };

    applyScroll();
    window.addEventListener(
        "scroll",
        () => {
            if (!scrollQueued) {
                scrollQueued = true;
                requestAnimationFrame(applyScroll);
            }
        },
        { passive: true }
    );

    if (reduceMotion || window.matchMedia("(pointer: coarse)").matches) return;

    let pointerQueued = false;
    let px = 0;
    let py = 0;

    window.addEventListener(
        "pointermove",
        (event) => {
            px = (event.clientX / window.innerWidth - 0.5) * 2;
            py = (event.clientY / window.innerHeight - 0.5) * 2;
            if (!pointerQueued) {
                pointerQueued = true;
                requestAnimationFrame(() => {
                    root.style.setProperty("--pointer-x", px.toFixed(3));
                    root.style.setProperty("--pointer-y", py.toFixed(3));
                    pointerQueued = false;
                });
            }
        },
        { passive: true }
    );
}

/* Reveal elements as they scroll into view, staggered by their order. */
function initReveal() {
    const targets = document.querySelectorAll("[data-reveal]");
    if (!targets.length) return;

    if (reduceMotion || !("IntersectionObserver" in window)) {
        targets.forEach((el) => el.classList.add("is-visible"));
        return;
    }

    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (!entry.isIntersecting) return;
                const index = Number(entry.target.dataset.revealIndex || 0);
                entry.target.style.setProperty("--reveal-delay", `${Math.min(index, 8) * 70}ms`);
                entry.target.classList.add("is-visible");
                observer.unobserve(entry.target);
            });
        },
        { rootMargin: "0px 0px -8% 0px", threshold: 0.12 }
    );

    let groupIndex = 0;
    let lastParent = null;
    targets.forEach((el) => {
        if (el.parentElement !== lastParent) {
            lastParent = el.parentElement;
            groupIndex = 0;
        }
        el.dataset.revealIndex = String(groupIndex++);
        observer.observe(el);
    });
}

/* Count numbers up once they are on screen. */
function initCounters() {
    const counters = document.querySelectorAll("[data-count-to]");
    if (!counters.length) return;

    const run = (el) => {
        const target = Number(el.dataset.countTo);
        const decimals = Number(el.dataset.countDecimals || 0);
        const suffix = el.dataset.countSuffix || "";
        if (!Number.isFinite(target)) return;

        if (reduceMotion) {
            el.textContent = target.toFixed(decimals) + suffix;
            return;
        }

        const duration = 1100;
        const start = performance.now();
        const tick = (now) => {
            const progress = Math.max(0, Math.min((now - start) / duration, 1));
            const eased = 1 - Math.pow(1 - progress, 3);
            el.textContent = (target * eased).toFixed(decimals) + suffix;
            if (progress < 1) requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
    };

    if (!("IntersectionObserver" in window)) {
        counters.forEach(run);
        return;
    }

    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (!entry.isIntersecting) return;
                run(entry.target);
                observer.unobserve(entry.target);
            });
        },
        { threshold: 0.4 }
    );
    counters.forEach((el) => observer.observe(el));
}

/* Grow progress bars from zero to their data-driven width. */
function initBars() {
    const bars = document.querySelectorAll("[data-bar-width]");
    if (!bars.length) return;

    const fill = (el) => {
        el.style.width = `${Math.max(0, Math.min(100, Number(el.dataset.barWidth)))}%`;
    };

    if (reduceMotion || !("IntersectionObserver" in window)) {
        bars.forEach(fill);
        return;
    }

    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (!entry.isIntersecting) return;
                requestAnimationFrame(() => fill(entry.target));
                observer.unobserve(entry.target);
            });
        },
        { threshold: 0.2 }
    );
    bars.forEach((el) => observer.observe(el));
}

/* Live countdown to lights out. */
function initCountdown() {
    const root = document.querySelector("[data-countdown]");
    if (!root) return;

    const target = new Date(root.dataset.countdown).getTime();
    if (!Number.isFinite(target)) return;

    const units = {
        days: root.querySelector('[data-unit="days"]'),
        hours: root.querySelector('[data-unit="hours"]'),
        minutes: root.querySelector('[data-unit="minutes"]'),
        seconds: root.querySelector('[data-unit="seconds"]'),
    };

    const pad = (value) => String(value).padStart(2, "0");

    const update = () => {
        const remaining = target - Date.now();
        if (remaining <= 0) {
            root.querySelectorAll("[data-unit]").forEach((el) => (el.textContent = "00"));
            return false;
        }
        const seconds = Math.floor(remaining / 1000);
        if (units.days) units.days.textContent = pad(Math.floor(seconds / 86400));
        if (units.hours) units.hours.textContent = pad(Math.floor((seconds % 86400) / 3600));
        if (units.minutes) units.minutes.textContent = pad(Math.floor((seconds % 3600) / 60));
        if (units.seconds) units.seconds.textContent = pad(seconds % 60);
        return true;
    };

    if (update()) {
        const timer = setInterval(() => {
            if (!update()) clearInterval(timer);
        }, 1000);
    }
}

function initNavToggle() {
    const rail = document.querySelector(".rail");
    const toggle = document.querySelector(".rail-toggle");
    if (!rail || !toggle) return;

    toggle.addEventListener("click", () => {
        const open = rail.classList.toggle("is-open");
        toggle.setAttribute("aria-expanded", String(open));
    });
}

/* --------------------------------------------------------------------------
   Season / Grand Prix / driver selectors
   -------------------------------------------------------------------------- */

function initRaceSelectors() {
    const yearSelect = document.getElementById("year");
    const roundSelect = document.getElementById("round");
    if (!yearSelect) return;

    const autoForm = document.querySelector("[data-autosubmit]");
    if (autoForm && !roundSelect) {
        yearSelect.addEventListener("change", () => autoForm.submit());
    }

    const raceForm = document.querySelector('[data-page="races"]');
    if (roundSelect && raceForm) {
        const params = new URLSearchParams(window.location.search);
        const preRound = params.get("round");
        yearSelect.addEventListener("change", () => loadRaces(yearSelect.value, null));

        const alreadyRendered = roundSelect.querySelector('option[value]:not([value=""])');
        if (!alreadyRendered && yearSelect.value) {
            loadRaces(yearSelect.value, preRound);
        }
    }

    const telemetryForm = document.querySelector('[data-page="telemetry"]');
    if (roundSelect && telemetryForm) {
        const preRound = window.__telemetryRound ?? null;
        const preDriver = window.__telemetryDriver ?? null;
        const driverSelect = document.getElementById("driver");

        yearSelect.addEventListener("change", () => {
            loadRaces(yearSelect.value, null);
            if (driverSelect) {
                driverSelect.innerHTML = '<option value="">Select a Grand Prix first…</option>';
            }
        });

        roundSelect.addEventListener("change", () => {
            if (yearSelect.value && roundSelect.value) {
                loadDrivers(yearSelect.value, roundSelect.value, null);
            }
        });

        if (yearSelect.value) {
            loadRaces(yearSelect.value, preRound).then(() => {
                if (preRound) loadDrivers(yearSelect.value, preRound, preDriver);
            });
        }
    }
}

async function loadRaces(year, selectRound) {
    const roundSelect = document.getElementById("round");
    if (!roundSelect || !year) return;

    roundSelect.innerHTML = '<option value="">Loading Grands Prix…</option>';
    roundSelect.disabled = true;
    roundSelect.classList.add("loading");

    try {
        const response = await fetch(`/get_races?year=${encodeURIComponent(year)}`);
        const data = await response.json();
        const races = data.races || data.rounds || [];

        roundSelect.innerHTML = "";
        roundSelect.disabled = false;
        roundSelect.classList.remove("loading");

        if (!races.length) {
            roundSelect.innerHTML = `<option value="">${
                data.error ? "Could not load the calendar" : "No races for this season"
            }</option>`;
            return;
        }

        const placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.textContent = "Select a Grand Prix…";
        placeholder.disabled = true;
        placeholder.selected = !selectRound;
        roundSelect.appendChild(placeholder);

        races.forEach((race) => {
            const option = document.createElement("option");
            option.value = race.round;
            option.textContent = race.country
                ? `R${race.round} · ${race.event_name} (${race.country})`
                : `R${race.round} · ${race.event_name}`;
            if (selectRound && String(race.round) === String(selectRound)) {
                option.selected = true;
                placeholder.selected = false;
            }
            roundSelect.appendChild(option);
        });
    } catch (error) {
        roundSelect.innerHTML = '<option value="">Could not load the calendar</option>';
        roundSelect.disabled = false;
        roundSelect.classList.remove("loading");
    }
}

async function loadDrivers(year, round, selectDriver) {
    const driverSelect = document.getElementById("driver");
    if (!driverSelect || !year || !round) return;

    driverSelect.innerHTML = '<option value="">Loading drivers…</option>';
    driverSelect.disabled = true;

    try {
        const response = await fetch(
            `/get_race_drivers?year=${encodeURIComponent(year)}&round=${encodeURIComponent(round)}`
        );
        const data = await response.json();
        driverSelect.innerHTML = '<option value="">Select a driver…</option>';
        driverSelect.disabled = false;

        (data.drivers || []).forEach((code) => {
            const option = document.createElement("option");
            option.value = code;
            option.textContent = code;
            if (selectDriver && code === selectDriver) option.selected = true;
            driverSelect.appendChild(option);
        });

        if (!data.drivers || !data.drivers.length) {
            driverSelect.innerHTML = '<option value="">No drivers found</option>';
        }
    } catch (error) {
        driverSelect.innerHTML = '<option value="">Could not load drivers</option>';
        driverSelect.disabled = false;
    }
}

/* Session data can take a few seconds to download, so say so on submit. */
function initFormFeedback() {
    document.querySelectorAll("form[data-pending]").forEach((form) => {
        form.addEventListener("submit", (event) => {
            const required = form.querySelectorAll("select[required]");
            for (const select of required) {
                if (!select.value) {
                    event.preventDefault();
                    select.focus();
                    return;
                }
            }

            const button = form.querySelector('button[type="submit"]');
            if (!button) return;
            button.disabled = true;
            button.innerHTML = `<span class="btn__spinner"></span>${form.dataset.pending}`;
        });
    });
}
