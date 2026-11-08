/*
 * The home page's screen: for each paradigm, one trial as GPT-2 read and answered it,
 * then the effects its run scored. Each scene paints one frame as a function of time,
 * from the numbers in paradigm-runs.js. Without JavaScript the Stroop frame shows instead.
 */

(() => {
    const WIDTH = 320;
    const HEIGHT = 312;
    const TEXT = "#f2f6ff";
    const MUTED = "#b8c5dc";
    const ACCENT = "#78a9f4";
    const BAR = "#4a6391";
    const TRACK = "rgba(242, 246, 255, 0.08)";
    const PANEL = "rgba(242, 246, 255, 0.06)";
    const RULE = "rgba(242, 246, 255, 0.32)";
    const FONT = "Geist, system-ui, sans-serif";
    const MONO = "\"Geist Mono\", ui-monospace, monospace";
    const STRONG = { weight: 600, color: TEXT };
    const SMALL = { size: 11 };
    const CODE = { size: 11, mono: true, color: TEXT };
    const AXIS = 170;
    const LINE = 14;
    const PROMPT_LINES = 3;
    const HOLD = 3;
    const ORDER = ["control", "memory", "learning", "language", "judgment"];
    const GROUPS = {
        control: "Cognitive control",
        memory: "Memory",
        learning: "Learning and transmission",
        language: "Language and semantics",
        judgment: "Judgment and reasoning"
    };
    const TITLES = {
        go_no_go: "Go/no-go",
        stop_signal: "Stop-signal",
        n_back: "N-back",
        drm: "DRM"
    };
    const MEASURES = {
        answer_surprisal: ["answer surprisal", "bits"],
        answer_margin: ["answer margin", "nats"],
        answer_probability: ["answer probability", null],
        option_probability: ["option probability", null],
        conjunction_probability: ["conjunction probability", null],
        expected_option: ["expected bin", null],
        recall: ["share recalled", null],
        correct: ["share correct", null],
        n_produced: ["words produced", null]
    };
    const READ = {
        span: (read, answer) => `Digits read: ${read.join(" ") || "none"}. Expected ${answer}.`,
        free_recall: (read) => `Studied words recalled: ${read.join(", ") || "none"}.`,
        fluency: (read) => `Words counted: ${read.join(", ") || "none"}.`
    };

    const measurer = document.createElement("canvas").getContext("2d");
    let ctx;

    const clamp = (value) => Math.min(1, Math.max(0, value));
    const span = (time, start, end) => clamp((time - start) / (end - start));
    const mix = (from, to, amount) => from + (to - from) * amount;

    function ease(time, start, end) {
        const amount = span(time, start, end);
        return amount * amount * (3 - 2 * amount);
    }

    function fontOf(style) {
        return `${style.weight || 400} ${style.size || 12}px ${style.mono ? MONO : FONT}`;
    }

    function measure(text, style = {}) {
        measurer.font = fontOf(style);
        return measurer.measureText(text).width;
    }

    // shortens text to a width, ending it with an ellipsis
    function fit(text, width, style = {}) {
        if (measure(text, style) <= width) {
            return text;
        }
        let cut = text;
        while (cut.length > 1 && measure(`${cut}…`, style) > width) {
            cut = cut.slice(0, -1);
        }
        return `${cut.trimEnd()}…`;
    }

    // shortens text to a width from its start, beginning it with an ellipsis
    function fitStart(text, width, style = {}) {
        let cut = text;
        while (cut.length > 1 && measure(`…${cut}`, style) > width) {
            cut = cut.slice(1);
        }
        return `…${cut.trimStart()}`;
    }

    // breaks text into lines no wider than `width`, keeping its own line breaks
    function wrap(text, width, style = {}) {
        const lines = [];
        text.split("\n").forEach((paragraph) => {
            let line = "";
            paragraph.split(" ").forEach((word) => {
                let rest = word;
                while (measure(rest, style) > width) {
                    let size = rest.length - 1;
                    while (size > 1 && measure(rest.slice(0, size), style) > width) {
                        size -= 1;
                    }
                    if (line) {
                        lines.push(line);
                        line = "";
                    }
                    lines.push(rest.slice(0, size));
                    rest = rest.slice(size);
                }
                const joined = line ? `${line} ${rest}` : rest;
                if (measure(joined, style) <= width) {
                    line = joined;
                } else {
                    lines.push(line);
                    line = rest;
                }
            });
            lines.push(line);
        });
        return lines;
    }

    function plain(value) {
        const size = Math.abs(value);
        let text = size.toFixed(2);
        if (size === 0) {
            text = "0";
        } else if (size < 0.01) {
            text = size.toExponential(1).replace("e-", "e−");
        } else if (size >= 100) {
            text = size.toFixed(0);
        }
        return value < 0 ? `−${text}` : text;
    }

    const signed = (value) => (value > 0 ? `+${plain(value)}` : plain(value));

    function probability(value) {
        if (value > 0 && value < 0.005) {
            return "<0.01";
        }
        return value > 0.995 && value < 1 ? ">0.99" : value.toFixed(2);
    }

    function verdict(effect) {
        if (!effect.detected) {
            return { word: "not found", color: MUTED };
        }
        return effect.difference > 0 ? { word: "found", color: ACCENT } : { word: "reversed", color: TEXT };
    }

    function fillBox(origin, size, color) {
        ctx.fillStyle = color;
        ctx.fillRect(
            origin[0],
            origin[1],
            size[0],
            size[1]
        );
    }

    function panel(origin, size, color) {
        ctx.beginPath();
        ctx.roundRect(
            origin[0],
            origin[1],
            size[0],
            size[1],
            6
        );
        ctx.fillStyle = color;
        ctx.fill();
    }

    function polyline(points, color, width = 1) {
        ctx.beginPath();
        points.forEach(([x, y], index) => {
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.stroke();
    }

    function disc(center, radius, color) {
        ctx.beginPath();
        ctx.arc(
            center[0],
            center[1],
            radius,
            0,
            Math.PI * 2
        );
        ctx.fillStyle = color;
        ctx.fill();
    }

    function label(text, position, style = {}) {
        ctx.font = fontOf(style);
        ctx.textAlign = style.align || "left";
        ctx.fillStyle = style.color || MUTED;
        ctx.fillText(text, position[0], position[1]);
    }

    // reveals the part of a figure to the left of a moving edge
    function revealed(box, amount, draw) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(
            box.left,
            box.top,
            box.width * amount,
            box.height
        );
        ctx.clip();
        draw();
        ctx.restore();
    }

    // marks the correct response
    function check(position, color) {
        const [x, y] = position;
        polyline([[x, y - 4], [x + 3, y - 1], [x + 8, y - 8]], color, 1.6);
    }

    // the end of what GPT-2 reads, in a panel, with the line just under it returned
    function promptLayout(text, top) {
        const lines = wrap(text, WIDTH - 16, CODE);
        const shown = lines.slice(-PROMPT_LINES);
        if (lines.length > PROMPT_LINES) {
            shown[0] = fitStart(shown[0], WIDTH - 16, CODE);
        }
        return {
            lines: shown,
            heading: lines.length > PROMPT_LINES ? "The end of what GPT-2 reads" : "What GPT-2 reads",
            top,
            bottom: top + shown.length * LINE + 10
        };
    }

    function drawPrompt(layout) {
        label(layout.heading, [0, layout.top - 7], SMALL);
        panel([0, layout.top], [WIDTH, layout.bottom - layout.top], PANEL);
        layout.lines.forEach((line, index) => {
            label(line, [8, layout.top + 16 + index * LINE], CODE);
        });
    }

    function optionsLayout(run, top) {
        const labels = run.options.map((option) => ({ text: option, answer: option === run.answer }));
        const widest = Math.max(...labels.map(({ text }) => measure(text, { weight: 600 })));
        const stacked = widest > 130;
        const start = stacked ? 0 : 14 + widest + 10;
        return {
            labels,
            stacked,
            start,
            width: WIDTH - start - (stacked ? 0 : 38),
            top,
            row: stacked ? 26 : 16,
            bottom: top + labels.length * (stacked ? 26 : 16)
        };
    }

    function drawOptions(run, layout, time) {
        const grown = ease(time, 0.4, 1.8);
        const shown = span(time, 1.8, 2.1);

        label("Probability of each response", [0, layout.top - 8], SMALL);
        layout.labels.forEach(({ text, answer }, index) => {
            const top = layout.top + index * layout.row;
            const value = run.probabilities[index];
            const name = layout.stacked ? fit(text, WIDTH - 60, {}) : text;
            const barTop = layout.stacked ? top + 15 : top + 4;
            const style = answer ? { weight: 600, color: ACCENT } : { color: TEXT };

            if (answer) {
                check([0, top + 11], ACCENT);
            }
            label(name, [14, top + 11], style);
            fillBox([layout.start, barTop], [layout.width, 7], TRACK);
            fillBox([layout.start, barTop], [Math.max(layout.width * value * grown, 0), 7], answer ? ACCENT : BAR);
            ctx.globalAlpha = shown;
            label(probability(value), [WIDTH, top + 11], { size: 11, color: TEXT, align: "right" });
            ctx.globalAlpha = 1;
        });
    }

    function writingLayout(run, top) {
        const lines = wrap(run.response, WIDTH - 16, CODE);
        const shown = lines.slice(0, 2);
        if (lines.length > 2) {
            shown[1] = fit(`${shown[1]} ${lines[2]}`, WIDTH - 16, CODE);
        }
        return {
            lines: shown,
            top,
            bottom: top + shown.length * LINE + 10 + 22
        };
    }

    function drawWriting(run, layout, time) {
        const total = layout.lines.join("").length;
        let typed = Math.floor(total * span(time, 0.4, 2.4));
        const panelBottom = layout.top + layout.lines.length * LINE + 10;

        label("What GPT-2 writes", [0, layout.top - 8], SMALL);
        panel([0, layout.top], [WIDTH, panelBottom - layout.top], PANEL);
        layout.lines.forEach((line, index) => {
            const part = line.slice(0, Math.max(typed, 0));
            const baseline = layout.top + 16 + index * LINE;
            label(part, [8, baseline], CODE);
            if (typed > 0 && typed < line.length) {
                fillBox([8 + measure(part, CODE) + 1, baseline - 9], [6, 11], ACCENT);
            }
            typed -= line.length;
        });
        ctx.globalAlpha = span(time, 2.4, 2.7);
        label(fit(run.read, WIDTH, {}), [0, panelBottom + 17], { color: TEXT });
        ctx.globalAlpha = 1;
    }

    // a level that ends in a number, such as "load 2", needs a colon before its mean
    const level = (name) => (/\d$/.test(name) ? `${name}:` : name);

    function effectLayout(effect) {
        const extent = Math.max(
            Math.abs(effect.low),
            Math.abs(effect.high),
            Math.abs(effect.difference)
        );
        const unit = effect.unit ? ` in ${effect.unit}` : "";
        const proxy = effect.proxy ? `, for ${effect.proxy}` : "";
        return {
            ...effect,
            extent: extent > 0 ? extent * 1.15 : 1,
            means: `${level(effect.a)} ${plain(effect.meanA)} vs ${level(effect.b)} ${plain(effect.meanB)}`,
            interval: `${signed(effect.difference)} [${plain(effect.low)}, ${plain(effect.high)}]`,
            about: `${effect.measure}${unit}${proxy}`,
            verdict: verdict(effect)
        };
    }

    function drawEffect(effect, top, start, time) {
        const moved = ease(time, start, start + 1);
        const opened = ease(time, start + 0.3, start + 1.1);
        const shown = span(time, start + 1, start + 1.3);
        const zero = AXIS / 2;
        const at = (value) => zero + (value / effect.extent) * zero;
        const point = mix(zero, at(effect.difference), moved);
        const low = mix(point, at(effect.low), opened);
        const high = mix(point, at(effect.high), opened);
        const middle = top + 22;
        const tint = effect.verdict.word === "not found" ? MUTED : effect.verdict.color;

        label(effect.name, [0, top + 12], STRONG);
        label(fit(effect.about, WIDTH, SMALL), [0, top + 52], SMALL);
        polyline([[0, middle], [AXIS, middle]], RULE);
        polyline([[zero, middle - 6], [zero, middle + 6]], RULE);
        if (moved > 0) {
            polyline([[low, middle], [high, middle]], tint, 2);
            polyline([[low, middle - 3], [low, middle + 3]], tint, 1.5);
            polyline([[high, middle - 3], [high, middle + 3]], tint, 1.5);
            disc([point, middle], 3.5, tint);
        }
        ctx.globalAlpha = shown;
        label(effect.verdict.word, [WIDTH, top + 12], { weight: 600, color: effect.verdict.color, align: "right" });
        label(effect.interval, [WIDTH, middle + 4], { size: 11, color: TEXT, align: "right" });
        label(fit(effect.means, WIDTH, SMALL), [0, top + 38], SMALL);
        ctx.globalAlpha = 1;
    }

    function summary(run, effects) {
        const found = effects.map((effect) => (
            `${effect.name} ${effect.verdict.word}, ${effect.interval} in ${effect.about}`
        ));
        return `${run.title} on GPT-2. ${run.task} ${found.join("; ")}.`;
    }

    function trialScene(run) {
        let layout = null;

        function arrange() {
            const prompt = promptLayout(run.prompt, 18);
            const below = prompt.bottom + 26;
            const answer = run.options ? optionsLayout(run, below) : writingLayout(run, below);
            const effects = run.effects.map(effectLayout);
            return {
                prompt,
                answer,
                effects,
                top: answer.bottom + 26,
                start: run.options ? 2.6 : 3
            };
        }

        return {
            title: run.title,
            duration: (run.options ? 2.6 : 3) + 1.6 * (run.effects.length - 1) + 1.4,
            label: () => summary(run, (layout || (layout = arrange())).effects),
            draw(time) {
                layout = layout || arrange();
                drawPrompt(layout.prompt);
                if (run.options) {
                    drawOptions(run, layout.answer, time);
                } else {
                    drawWriting(run, layout.answer, time);
                }
                label("Scored over the whole run", [0, layout.top - 8], SMALL);
                layout.effects.forEach((effect, index) => {
                    drawEffect(effect, layout.top + index * 60, layout.start + index * 1.6, time);
                });
            }
        };
    }

    // agreement with the previous generation and with the seed, over the generations
    function chainScene(run) {
        const box = { left: 46, top: 124, width: 176, height: 112 };
        const reach = { left: 0, top: 0, width: box.left + box.width + 4, height: HEIGHT };
        const ends = box.left + box.width + 10;
        const axisLabels = box.top + box.height + 15;
        const y = (value) => box.top + box.height * (1 - (value - 0.4) / 0.6);
        const x = (index, count) => box.left + (box.width * index) / (count - 1);
        const { fidelity, accuracy } = run.series;
        const last = fidelity.length - 1;
        let prompt = null;

        function line(values, color, width) {
            polyline(values.map((value, index) => [x(index, values.length), y(value)]), color, width);
        }

        return {
            title: run.title,
            duration: 4.6,
            label: () => (
                `${run.title} on GPT-2, ${run.chains} chains over ${fidelity.length} generations. `
                + "Agreement with the previous "
                + `generation rises from ${plain(fidelity[0])} to ${plain(fidelity[last])}, and agreement `
                + `with the seed stays near chance, at ${plain(accuracy[last])}.`
            ),
            draw(time) {
                prompt = prompt || promptLayout(run.prompt, 18);
                const drawn = ease(time, 0.4, 3.2);
                const labelled = span(time, 3.2, 3.5);

                drawPrompt(prompt);
                label(`Agreement in ${run.chains} chains, generation by generation`, [0, prompt.bottom + 18], SMALL);
                ctx.setLineDash([3, 3]);
                polyline([[box.left, y(0.5)], [box.left + box.width, y(0.5)]], RULE);
                ctx.setLineDash([]);
                polyline([[box.left, box.top + box.height], [box.left + box.width, box.top + box.height]], RULE);
                label("1.0", [box.left - 6, y(1) + 4], { size: 11, align: "right" });
                label("0.5", [box.left - 6, y(0.5) + 4], { size: 11, align: "right" });
                label("chance", [box.left - 6, y(0.5) + 17], { size: 11, align: "right" });
                label("1", [x(0, fidelity.length), axisLabels], { size: 11, align: "center" });
                label(String(fidelity.length), [x(last, fidelity.length), axisLabels], { size: 11, align: "center" });
                label("generation", [box.left + box.width / 2, axisLabels], { size: 11, align: "center" });
                revealed(reach, drawn, () => {
                    line(accuracy, MUTED, 1.5);
                    line(fidelity, ACCENT, 2);
                });
                ctx.globalAlpha = labelled;
                label(`${plain(fidelity[last])} with the`, [ends, y(fidelity[last]) + 4], { ...STRONG, color: ACCENT });
                label("previous one", [ends, y(fidelity[last]) + 18], { color: ACCENT });
                label(`${plain(accuracy[last])} with`, [ends, y(accuracy[last]) + 18], { color: TEXT });
                label("the seed", [ends, y(accuracy[last]) + 32], { color: TEXT });
                ctx.globalAlpha = ease(time, 3.7, 4.2);
                label("Each generation copies the last more faithfully,", [0, HEIGHT - 26], STRONG);
                label("while agreement with the seed stays at chance.", [0, HEIGHT - 10], STRONG);
                ctx.globalAlpha = 1;
            }
        };
    }

    const capitalized = (text) => text.charAt(0).toUpperCase() + text.slice(1);

    // a level that is only a number takes the name of what it measures, as in "load 2"
    function named(value, by) {
        const words = String(value).replaceAll("_", " ");
        if (/^\d+(\.\d+)?$/.test(words) && by) {
            return `${by.replaceAll("_", " ")} ${words}`;
        }
        return words === "all" ? "all trials" : words;
    }

    function scored(row) {
        const [measure, unit] = MEASURES[row.measure] || [row.measure.replaceAll("_", " "), null];
        return {
            name: row.effect.replaceAll("_", " "),
            measure,
            unit,
            proxy: row.proxy_for,
            a: named(row.level_a, row.by),
            b: named(row.level_b, row.by),
            meanA: row.mean_a,
            meanB: row.mean_b,
            difference: row.difference,
            low: row.ci_low,
            high: row.ci_high,
            detected: row.detected
        };
    }

    // turns a run as the tour wrote it into the words and numbers the screen shows
    function present(raw) {
        const run = {
            ...raw,
            title: TITLES[raw.name] || capitalized(raw.name.replaceAll("_", " ")),
            group: GROUPS[raw.page],
            family: capitalized(raw.page),
            href: `paradigms/${raw.page}/#${raw.anchor}`
        };
        if (raw.response !== undefined) {
            run.response = raw.response.split(/\s+/).join(" ").trim();
            const reader = READ[raw.name] || ((read) => `Read: ${read.join(", ") || "nothing"}.`);
            run.read = reader(raw.read, raw.answer);
        }
        if (raw.effects) {
            run.effects = raw.effects.map(scored);
        }
        return run;
    }

    const RUNS = (window.COGNARIUM_RUNS || [])
        .map(present)
        .sort((first, second) => ORDER.indexOf(first.page) - ORDER.indexOf(second.page));
    const SCENES = Object.fromEntries(RUNS.map((run) => [run.name, run.series ? chainScene(run) : trialScene(run)]));
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function createPlayer(figure, hooks = {}) {
        const canvas = figure.querySelector("canvas");
        const context = canvas.getContext("2d");
        let scene = SCENES[figure.dataset.figure];
        let time = reducedMotion ? scene.duration : 0;
        let frame = 0;
        let stamp = null;
        let state = "idle";
        let control = null;

        function paintAt(moment) {
            const scale = canvas.width / WIDTH;

            ctx = context;
            ctx.setTransform(
                scale,
                0,
                0,
                scale,
                0,
                0
            );
            ctx.clearRect(
                0,
                0,
                WIDTH,
                HEIGHT
            );
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
            scene.draw(moment);
        }

        function fitCanvas() {
            const width = canvas.getBoundingClientRect().width;
            const density = Math.min(window.devicePixelRatio || 1, 3);

            if (width > 0) {
                canvas.width = Math.round(width * density);
                canvas.height = Math.round(canvas.width * (HEIGHT / WIDTH));
                paintAt(time);
            }
        }

        function setState(next) {
            state = next;

            if (control) {
                const word = { playing: "Pause", done: "Replay" }[next] || "Play";
                control.textContent = word;
                control.setAttribute("aria-label", `${word} the run of ${scene.title}`);
            }
            if (next === "done" && hooks.onDone) {
                hooks.onDone();
            }
        }

        function advance(now) {
            const step = stamp === null ? 0 : Math.min(now - stamp, 100) / 1000;

            stamp = now;
            time = Math.min(time + step, scene.duration);
            paintAt(time);

            if (time < scene.duration) {
                frame = requestAnimationFrame(advance);
            } else {
                setState("done");
            }
        }

        function start() {
            stamp = null;
            setState("playing");
            frame = requestAnimationFrame(advance);
        }

        function stop() {
            cancelAnimationFrame(frame);
        }

        function restart() {
            stop();
            time = 0;
            start();
        }

        function toggle() {
            if (state === "playing") {
                stop();
                setState("paused");
                hooks.onPause?.();
            } else if (state === "paused") {
                start();
            } else {
                restart();
            }
        }

        function describe() {
            canvas.setAttribute("aria-label", scene.label());
        }

        // shows another scene from its start, playing it if the figure is in view
        function show(name, play) {
            stop();
            scene = SCENES[name];
            figure.dataset.figure = name;
            time = reducedMotion ? scene.duration : 0;
            describe();
            paintAt(time);
            if (play && !reducedMotion) {
                start();
            } else {
                setState(reducedMotion ? "done" : "ready");
            }
        }

        // a figure the reader paused stays paused; any other replays when it comes back into view
        function onVisibility({ isIntersecting, intersectionRatio }) {
            if (intersectionRatio >= 0.4 && (state === "idle" || state === "ready")) {
                restart();
            }

            if (!isIntersecting && state !== "paused") {
                stop();
                time = 0;
                setState("ready");
                paintAt(time);
            }
        }

        new ResizeObserver(fitCanvas).observe(canvas);
        describe();
        fitCanvas();

        if (!reducedMotion) {
            control = document.createElement("button");
            control.type = "button";
            control.className = "figure-control";
            figure.append(control);
            control.addEventListener("click", toggle);
            canvas.addEventListener("click", toggle);
            setState("idle");
        }

        return {
            figure,
            show,
            onVisibility,
            finished: () => state === "done",
            paintFinal: () => paintAt(scene.duration)
        };
    }

    // the reel steps through every scene once, and stops for good on any choice the reader makes
    function createReel(reel, observe) {
        const figure = reel.querySelector("[data-figure]");
        const names = RUNS.map((run) => run.name);
        const about = {
            group: reel.querySelector(".cg-reel__group"),
            title: reel.querySelector(".cg-reel__title a"),
            task: reel.querySelector(".cg-reel__task")
        };
        const caption = reel.querySelector(".cg-reel__about");
        let index = Math.max(names.indexOf(figure.dataset.figure), 0);
        let rolling = !reducedMotion;
        let visible = false;
        let wait = 0;
        let counter = null;
        let ticks = [];

        function hold() {
            cancelAnimationFrame(wait);
            let waited = 0;
            let last = null;
            const step = (now) => {
                waited += last === null ? 0 : Math.min(now - last, 100) / 1000;
                last = now;
                if (!rolling || !visible) {
                    return;
                }
                if (waited >= HOLD) {
                    go(index + 1, true);
                } else {
                    wait = requestAnimationFrame(step);
                }
            };
            if (rolling && index < names.length - 1) {
                wait = requestAnimationFrame(step);
            }
        }

        const player = createPlayer(figure, {
            onDone: hold,
            onPause: () => settle()
        });

        // the reader has taken over, so the reel stops moving on and starts announcing changes
        function settle() {
            rolling = false;
            cancelAnimationFrame(wait);
            caption.setAttribute("aria-live", "polite");
        }

        function go(next, play) {
            const run = RUNS[(next + names.length) % names.length];
            cancelAnimationFrame(wait);
            index = names.indexOf(run.name);
            about.group.textContent = run.group;
            about.title.textContent = run.title;
            about.title.setAttribute("href", run.href);
            about.task.textContent = run.task;
            counter.textContent = `${index + 1} of ${names.length}`;
            ticks.forEach((tick, position) => {
                if (position === index) {
                    tick.setAttribute("aria-current", "true");
                } else {
                    tick.removeAttribute("aria-current");
                }
            });
            player.show(run.name, play);
        }

        function choose(next) {
            settle();
            go(next, visible);
        }

        function button(text, name, onClick) {
            const element = document.createElement("button");
            element.type = "button";
            element.className = name;
            element.textContent = text;
            element.setAttribute("aria-controls", figure.id);
            element.addEventListener("click", onClick);
            return element;
        }

        function buildNav() {
            const nav = document.createElement("div");
            const steps = document.createElement("div");
            const track = document.createElement("div");
            const groups = new Map();

            nav.className = "cg-reel__nav";
            steps.className = "cg-reel__steps";
            track.className = "cg-reel__ticks";
            counter = document.createElement("span");
            counter.className = "cg-reel__count";
            steps.append(
                button("Previous", "cg-reel__step", () => choose(index - 1)),
                counter,
                button("Next", "cg-reel__step", () => choose(index + 1))
            );
            RUNS.forEach((run, position) => {
                if (!groups.has(run.group)) {
                    const group = document.createElement("div");
                    const name = document.createElement("span");
                    group.className = "cg-reel__group-ticks";
                    name.textContent = run.family;
                    name.className = "cg-reel__group-name";
                    group.append(document.createElement("div"), name);
                    groups.set(run.group, group.firstChild);
                    track.append(group);
                }
                const tick = button("", "cg-reel__tick", () => choose(position));
                tick.setAttribute("aria-label", run.title);
                groups.get(run.group).append(tick);
                ticks.push(tick);
            });
            nav.append(steps, track);
            caption.after(nav);
            reel.addEventListener("keydown", (event) => {
                const inside = event.target.closest(".cg-reel__screen, .cg-reel__about, .cg-reel__nav");
                if (inside && (event.key === "ArrowLeft" || event.key === "ArrowRight")) {
                    event.preventDefault();
                    choose(index + (event.key === "ArrowLeft" ? -1 : 1));
                }
            });
        }

        buildNav();
        if (reducedMotion) {
            caption.setAttribute("aria-live", "polite");
        }
        go(index, false);
        reel.classList.add("cg-reel--ready");
        observe(figure, (entry) => {
            visible = entry.intersectionRatio >= 0.4;
            player.onVisibility(entry);
            if (visible && player.finished() && rolling) {
                hold();
            }
        });
        return player;
    }

    function init() {
        const players = [];
        const watchers = new Map();
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => watchers.get(entry.target)?.(entry));
        }, { threshold: [0, 0.4] });
        const observe = (figure, watch) => {
            watchers.set(figure, watch);
            if (!reducedMotion) {
                observer.observe(figure);
            }
        };

        document.querySelectorAll(".cg-reel").forEach((reel) => {
            players.push(createReel(reel, observe));
        });
        document.querySelectorAll("[data-figure]").forEach((figure) => {
            if (!figure.closest(".cg-reel") && SCENES[figure.dataset.figure]) {
                const player = createPlayer(figure);
                players.push(player);
                observe(figure, player.onVisibility);
            }
        });
        window.addEventListener("beforeprint", () => {
            players.forEach(({ paintFinal }) => paintFinal());
        });
    }

    // canvas text does not wait for webfonts the way page text does
    Promise.all([
        document.fonts.load(`400 12px ${FONT}`),
        document.fonts.load(`600 12px ${FONT}`),
        document.fonts.load(`400 11px ${MONO}`)
    ]).then(init, init);
})();
