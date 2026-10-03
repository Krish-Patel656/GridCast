/* Race playback.

   The server sends each driver's cumulative lap times and the circuit outline.
   Everything on screen is derived from a single simulated clock: where a car
   sits on the track, the running order, and the gaps. Nothing is pre-baked per
   frame, so scrubbing backwards is just as cheap as playing forwards.

   Track points are sampled at even intervals of *time* around a real lap, so
   stepping through them at a constant rate makes the cars brake into corners
   and stretch their legs on the straights without any extra work. */

(() => {
    const dataEl = document.getElementById("sim-data");
    const svg = document.querySelector(".sim__track");
    if (!dataEl || !svg) return;

    const data = JSON.parse(dataEl.textContent);
    const points = data.track || [];
    const drivers = data.drivers || [];
    if (points.length < 3 || !drivers.length) return;

    const laps = data.laps;
    const leader = drivers[0];

    /* 1x plays the whole race in about two minutes. */
    const BASE_COMPRESSION = data.raceTime / 120;

    const state = {
        time: 0,
        speed: 2,
        playing: false,
        lastFrame: 0,
    };

    /* ---------------------------------------------------------------- track */

    const trackPath = points.map(([x, y], i) => `${i ? "L" : "M"}${x} ${y}`).join(" ") + " Z";
    svg.querySelectorAll("[data-track-path]").forEach((path) => path.setAttribute("d", trackPath));

    /* Circuits have wildly different proportions, so the canvas is cropped to
       this one's bounding box rather than a fixed square. */
    const xs = points.map((p) => p[0]);
    const ys = points.map((p) => p[1]);
    const pad = 46;
    svg.setAttribute(
        "viewBox",
        [
            Math.min(...xs) - pad,
            Math.min(...ys) - pad,
            Math.max(...xs) - Math.min(...xs) + pad * 2,
            Math.max(...ys) - Math.min(...ys) + pad * 2,
        ].join(" ")
    );

    const startLine = svg.querySelector("[data-start-line]");
    if (startLine) {
        const [sx, sy] = points[0];
        const [nx, ny] = points[Math.min(6, points.length - 1)];
        const angle = Math.atan2(ny - sy, nx - sx) + Math.PI / 2;
        const reach = 17;
        startLine.innerHTML =
            `<line x1="${sx - Math.cos(angle) * reach}" y1="${sy - Math.sin(angle) * reach}" ` +
            `x2="${sx + Math.cos(angle) * reach}" y2="${sy + Math.sin(angle) * reach}" ` +
            `stroke="rgba(255,255,255,0.85)" stroke-width="4" stroke-dasharray="4 3"/>`;
    }

    const pointAt = (fraction) => {
        const wrapped = ((fraction % 1) + 1) % 1;
        const exact = wrapped * (points.length - 1);
        const i = Math.floor(exact);
        const t = exact - i;
        const a = points[i];
        const b = points[Math.min(i + 1, points.length - 1)];
        return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    };

    /* ----------------------------------------------------------------- cars */

    const carLayer = svg.querySelector("[data-cars]");
    const board = document.querySelector("[data-sim-board]");

    const cars = drivers.map((driver) => {
        const group = document.createElementNS(svg.namespaceURI, "g");
        group.setAttribute("class", "sim-car");
        group.innerHTML =
            `<circle r="13" fill="${driver.team_color}" stroke="rgba(0,0,0,0.65)" stroke-width="2.5"/>` +
            `<text y="-20" text-anchor="middle" class="sim-car__label">${driver.driver_code}</text>`;
        carLayer.appendChild(group);

        const row = document.createElement("li");
        row.className = "sim__row";
        row.innerHTML =
            `<span class="sim__pos"></span>` +
            `<span class="sim__code" style="color:${driver.team_color}">${driver.driver_code}</span>` +
            `<span class="sim__bar" style="background:${driver.team_color}"></span>` +
            `<span class="sim__gap"></span>`;
        board.appendChild(row);

        return {
            driver,
            group,
            row,
            label: group.querySelector("text"),
            posEl: row.querySelector(".sim__pos"),
            gapEl: row.querySelector(".sim__gap"),
        };
    });

    /* Three-letter codes smear into an unreadable smudge when the field is nose
       to tail, so a label is drawn only if it clears the ones already placed.
       Labels are wide and short, hence the elliptical test, and offering the
       underside as a second slot lets roughly twice as many through. */
    const LABEL_SLOTS = [-20, 32];
    const LABEL_REACH_X = 44;
    const LABEL_REACH_Y = 24;

    const placeLabels = (running) => {
        const placed = [];
        running.forEach((car) => {
            const slot = LABEL_SLOTS.find((dy) =>
                placed.every(([px, py]) => {
                    const ex = (car.x - px) / LABEL_REACH_X;
                    const ey = (car.y + dy - py) / LABEL_REACH_Y;
                    return ex * ex + ey * ey >= 1;
                })
            );
            if (slot === undefined) {
                car.label.setAttribute("opacity", "0");
                return;
            }
            placed.push([car.x, car.y + slot]);
            car.label.setAttribute("y", String(slot));
            car.label.setAttribute("opacity", "1");
        });
    };

    /* Every car's clock starts at zero, so without this they would all be
       stacked on the start line at lights out. Cars are drawn back along the
       track by their grid slot, and the stagger closes over the opening lap
       the way a real field spreads out of the first corner. */
    const GRID_STAGGER = 0.004;

    const trackProgress = (car, progress) => {
        const slot = (car.driver.grid || 1) - 1;
        const closing = Math.max(0, 1 - progress);
        return progress - slot * GRID_STAGGER * closing;
    };

    /* Lap completed plus the fraction of the current lap, from the clock. */
    const progressAt = (driver, time) => {
        const cum = driver.cumulative;
        if (time >= cum[cum.length - 1]) return laps;

        let lap = 0;
        while (lap < cum.length && cum[lap] <= time) lap += 1;
        const start = lap === 0 ? 0 : cum[lap - 1];
        const end = cum[lap];
        return lap + (end > start ? (time - start) / (end - start) : 0);
    };

    /* When did this driver reach that point on track? Inverse of the above,
       which is what turns a distance gap into the time gap a timing screen
       would show. */
    const timeAtProgress = (driver, progress) => {
        const cum = driver.cumulative;
        const capped = Math.min(progress, laps);
        const lap = Math.min(Math.floor(capped), cum.length - 1);
        const frac = capped - lap;
        const start = lap === 0 ? 0 : cum[lap - 1];
        const end = cum[lap];
        return start + frac * (end - start);
    };

    const formatClock = (seconds) => {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${String(s).padStart(2, "0")}`;
    };

    const formatGap = (car, progress, running) => {
        const behind = running[0];
        if (car.driver === behind.driver) return "leader";

        const lapsDown = behind.progress - progress;
        if (lapsDown >= 1) {
            const whole = Math.floor(lapsDown);
            return `+${whole} lap${whole > 1 ? "s" : ""}`;
        }
        const gap = state.time - timeAtProgress(behind.driver, progress);
        return `+${gap.toFixed(1)}s`;
    };

    const lapEl = document.querySelector("[data-sim-lap]");
    const clockEl = document.querySelector("[data-sim-clock]");
    const leaderEl = document.querySelector("[data-sim-leader]");
    const scrub = document.querySelector("[data-sim-scrub]");

    const render = () => {
        /* Sorting on the staggered value keeps the running order and the cars
           on screen telling the same story, including on the grid. */
        const running = cars
            .map((car) => {
                const progress = progressAt(car.driver, state.time);
                return { ...car, progress, shown: trackProgress(car, progress) };
            })
            .sort((a, b) => b.shown - a.shown);

        running.forEach((car, index) => {
            const [x, y] = pointAt(car.shown);
            car.x = x;
            car.y = y;
            car.group.setAttribute("transform", `translate(${x} ${y})`);
            car.group.classList.toggle("is-leader", index === 0);

            const lapNumber = Math.min(Math.floor(car.progress) + 1, laps);
            const inPit = car.driver.pit_laps.includes(lapNumber);
            car.group.classList.toggle("is-pitting", inPit);

            car.posEl.textContent = index + 1;
            car.gapEl.textContent = inPit ? "pit" : formatGap(car, car.progress, running);
            car.row.classList.toggle("is-pitting", inPit);
            car.row.classList.toggle("is-leader", index === 0);
            car.row.classList.toggle("is-podium", index > 0 && index < 3);
            car.row.style.order = String(index);
        });

        placeLabels(running);

        const leaderProgress = running[0].progress;
        if (lapEl) lapEl.textContent = `${Math.min(Math.floor(leaderProgress) + 1, laps)} / ${laps}`;
        if (clockEl) clockEl.textContent = formatClock(state.time);
        if (leaderEl) leaderEl.textContent = running[0].driver.driver_code;
        if (scrub && document.activeElement !== scrub) {
            scrub.value = String(Math.min(leaderProgress + 1, laps));
        }
    };

    /* ------------------------------------------------------------- playback */

    const finished = () => state.time >= leader.cumulative[leader.cumulative.length - 1];

    const toggle = document.querySelector("[data-sim-toggle]");
    const toggleText = document.querySelector("[data-sim-toggle-text]");
    const iconPlay = document.querySelector("[data-icon-play]");
    const iconPause = document.querySelector("[data-icon-pause]");

    const setPlaying = (playing) => {
        state.playing = playing;
        if (iconPlay) iconPlay.hidden = playing;
        if (iconPause) iconPause.hidden = !playing;
        if (toggleText) {
            toggleText.textContent = playing ? "Pause" : finished() ? "Replay" : "Start race";
        }
        if (playing) {
            state.lastFrame = performance.now();
            requestAnimationFrame(step);
        }
    };

    const step = (now) => {
        if (!state.playing) return;
        const elapsed = (now - state.lastFrame) / 1000;
        state.lastFrame = now;
        state.time += elapsed * state.speed * BASE_COMPRESSION;

        if (finished()) {
            state.time = leader.cumulative[leader.cumulative.length - 1];
            render();
            setPlaying(false);
            return;
        }
        render();
        requestAnimationFrame(step);
    };

    if (toggle) {
        toggle.addEventListener("click", () => {
            if (finished()) state.time = 0;
            setPlaying(!state.playing);
        });
    }

    document.querySelectorAll("[data-sim-speed]").forEach((button) => {
        button.addEventListener("click", () => {
            state.speed = Number(button.dataset.simSpeed);
            document
                .querySelectorAll("[data-sim-speed]")
                .forEach((other) => other.classList.toggle("is-active", other === button));
        });
    });

    if (scrub) {
        const seek = () => {
            state.time = timeAtProgress(leader, Number(scrub.value) - 1);
            render();
        };
        scrub.addEventListener("input", () => {
            setPlaying(false);
            seek();
        });
    }

    /* Space bar is the obvious key for play/pause once the stage is in view. */
    document.addEventListener("keydown", (event) => {
        if (event.code !== "Space" || event.target.matches("input, select, textarea, button")) return;
        const stage = svg.getBoundingClientRect();
        if (stage.bottom < 0 || stage.top > window.innerHeight) return;
        event.preventDefault();
        if (finished()) state.time = 0;
        setPlaying(!state.playing);
    });

    render();

    /* Roll the start once the track is actually on screen. */
    if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (!entry.isIntersecting) return;
                    observer.disconnect();
                    setPlaying(true);
                });
            },
            { threshold: 0.3 }
        );
        observer.observe(svg);
    }
})();
