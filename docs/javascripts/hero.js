/*
 * Home page art: a layered network whose units turn into neurons from left to right.
 *
 * Spikes travel along the connections. Each unit sums the spikes it receives, leaks
 * toward rest, and fires when it crosses a threshold, so activity spreads as waves.
 * The structure is drawn once on one canvas and the activity every frame on another.
 */
(() => {
  "use strict";

  const LAYERS = [6, 8, 9, 9, 8, 7, 5];
  const THRESHOLD = 1;
  const LEAK = 0.9;
  const REFRACTORY = 0.45;
  const SPEED = 140;
  const TAIL = 30;
  const MAX_SPIKES = 150;
  const VOLLEY_EVERY = [3.2, 5.2];

  /** Returns a seeded random number generator (mulberry32), so the layout is stable. */
  function seeded(seed) {
    return () => {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function smoothstep(low, high, x) {
    const t = Math.min(1, Math.max(0, (x - low) / (high - low)));
    return t * t * (3 - 2 * t);
  }

  function rgb(hex) {
    const value = hex.replace("#", "");
    const full = value.length === 3 ? value.replace(/./g, "$&$&") : value;
    const number = parseInt(full, 16);
    return [(number >> 16) & 255, (number >> 8) & 255, number & 255];
  }

  function rgba(colour, alpha) {
    return `rgba(${colour[0]}, ${colour[1]}, ${colour[2]}, ${alpha})`;
  }

  function mix(a, b, t) {
    return a.map((channel, index) => Math.round(channel + (b[index] - channel) * t));
  }

  /** Reads the colours for the current light or dark scheme from CSS custom properties. */
  function palette(element) {
    const style = getComputedStyle(element);
    const read = (name, fallback) => style.getPropertyValue(name).trim() || fallback;
    return {
      line: rgb(read("--cg-art-line", "#2856d8")),
      soma: rgb(read("--cg-art-soma", "#2856d8")),
      spike: rgb(read("--cg-art-spike", "#2856d8")),
      glow: rgb(read("--cg-art-glow", "#78a9f4")),
      ground: rgb(read("--cg-art-ground", "#ffffff")),
      lineAlpha: parseFloat(read("--cg-art-line-alpha", "0.2")),
      blend: read("--cg-art-blend", "source-over")
    };
  }

  /** Samples a cubic Bézier into a polyline with cumulative lengths, for spikes to follow. */
  function sample(p0, p1, p2, p3) {
    const steps = 36;
    const points = new Float32Array((steps + 1) * 2);
    const lengths = new Float32Array(steps + 1);
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const u = 1 - t;
      const x = u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0];
      const y = u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1];
      points[i * 2] = x;
      points[i * 2 + 1] = y;
      if (i > 0) {
        lengths[i] = lengths[i - 1] + Math.hypot(x - points[i * 2 - 2], y - points[i * 2 - 1]);
      }
    }
    return { points, lengths, length: lengths[steps] };
  }

  /** Finds the point a given distance along a sampled path. */
  function along(path, distance) {
    const { points, lengths } = path;
    let low = 0;
    let high = lengths.length - 1;
    while (high - low > 1) {
      const middle = (low + high) >> 1;
      if (lengths[middle] < distance) {
        low = middle;
      } else {
        high = middle;
      }
    }
    const span = lengths[high] - lengths[low] || 1;
    const t = Math.min(1, Math.max(0, (distance - lengths[low]) / span));
    return [
      points[low * 2] + (points[high * 2] - points[low * 2]) * t,
      points[low * 2 + 1] + (points[high * 2 + 1] - points[low * 2 + 1]) * t
    ];
  }

  /** Grows a dendrite as tapering segments that fork, more often the more biological the unit. */
  function grow(segments, x, y, angle, length, width, depth, random) {
    const steps = 4;
    let cx = x;
    let cy = y;
    let heading = angle;
    for (let step = 0; step < steps; step++) {
      heading += (random() - 0.5) * 0.55;
      const nx = cx + (Math.cos(heading) * length) / steps;
      const ny = cy + (Math.sin(heading) * length) / steps;
      segments.push([cx, cy, nx, ny, width * (1 - (0.45 * step) / steps)]);
      cx = nx;
      cy = ny;
    }
    if (depth > 0) {
      const fork = 0.32 + random() * 0.36;
      grow(segments, cx, cy, heading - fork, length * 0.62, width * 0.55, depth - 1, random);
      grow(segments, cx, cy, heading + fork, length * 0.62, width * 0.55, depth - 1, random);
    }
  }

  /** Lays out units in layers, then connects them, jittering and bending more to the right. */
  function build(width, height) {
    const random = seeded(11);
    const units = [];
    const layers = LAYERS.map((count, layer) => {
      const fraction = 0.2 + (layer * 0.7) / (LAYERS.length - 1);
      const spread = Math.min(height * 0.6, count * 50);
      const ids = [];
      for (let index = 0; index < count; index++) {
        const bio = smoothstep(0.36, 0.86, fraction);
        const offset = count === 1 ? 0 : index / (count - 1) - 0.5;
        const x = fraction * width + (random() - 0.5) * width * 0.07 * bio;
        const y = height * 0.43 + offset * spread + (random() - 0.5) * (spread / count) * 1.4 * bio;
        const b = smoothstep(0.36, 0.86, x / width);
        const unit = {
          x,
          y,
          b,
          layer,
          radius: 3.6 + 1.6 * b,
          phase: random() * Math.PI * 2,
          potential: 0,
          fired: -10,
          flash: 0,
          rate: layer === 0 ? 0.35 : 0.02,
          out: [],
          dendrites: []
        };
        if (b > 0.18) {
          const branches = 3 + Math.floor(random() * 3);
          const depth = b > 0.62 ? 2 : 1;
          for (let d = 0; d < branches; d++) {
            const angle = d === 0 ? -Math.PI / 2 + (random() - 0.5) * 0.5 : Math.PI + (random() - 0.5) * Math.PI * 1.5;
            const length = (10 + random() * 16) * (0.35 + b);
            grow(unit.dendrites, x, y, angle, length, 1.5 * b + 0.3, depth, random);
          }
        }
        units.push(unit);
        ids.push(units.length - 1);
      }
      return ids;
    });

    const edges = [];
    const connect = (from, to, arc) => {
      const a = units[from];
      const c = units[to];
      const dx = c.x - a.x;
      const dy = c.y - a.y;
      const length = Math.hypot(dx, dy) || 1;
      const nx = -dy / length;
      const ny = dx / length;
      const bend = (a.b + c.b) / 2;
      const swing = arc ? 0.45 : 0.26;
      const k1 = (random() - 0.5) * 2 * bend * length * swing + (arc ? length * 0.3 : 0);
      const k2 = (random() - 0.5) * 2 * bend * length * swing + (arc ? length * 0.3 : 0);
      const path = sample(
        [a.x, a.y],
        [a.x + dx * 0.33 + nx * k1, a.y + dy * 0.33 + ny * k1],
        [a.x + dx * 0.66 + nx * k2, a.y + dy * 0.66 + ny * k2],
        [c.x, c.y]
      );
      const edge = { from, to, path, bend, weight: 0.38 + random() * 0.3, arc };
      a.out.push(edge);
      edges.push(edge);
    };

    for (let layer = 0; layer < layers.length - 1; layer++) {
      const sources = layers[layer];
      const targets = layers[layer + 1];
      const inputs = new Map(targets.map((id) => [id, 0]));
      for (const from of sources) {
        let made = 0;
        for (const to of targets) {
          if (random() < 0.4) {
            connect(from, to, false);
            inputs.set(to, inputs.get(to) + 1);
            made++;
          }
        }
        if (made === 0) {
          const to = targets[Math.floor(random() * targets.length)];
          connect(from, to, false);
          inputs.set(to, inputs.get(to) + 1);
        }
      }
      for (const [to, count] of inputs) {
        for (let extra = count; extra < 2; extra++) {
          connect(sources[Math.floor(random() * sources.length)], to, false);
        }
      }
    }

    for (const unit of units) {
      if (unit.b < 0.6 || random() > 0.5) {
        continue;
      }
      const near = units.filter(
        (other) => other !== unit && other.layer <= unit.layer && other.b > 0.45 && Math.hypot(other.x - unit.x, other.y - unit.y) < 130
      );
      if (near.length) {
        connect(units.indexOf(unit), units.indexOf(near[Math.floor(random() * near.length)]), true);
      }
    }

    return { units, layers, edges };
  }

  /** Makes a soft round glow, drawn once and stamped for every spike and flash. */
  function sprite(colour) {
    const size = 64;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const context = canvas.getContext("2d");
    const gradient = context.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    gradient.addColorStop(0, rgba(colour, 1));
    gradient.addColorStop(0.25, rgba(colour, 0.45));
    gradient.addColorStop(1, rgba(colour, 0));
    context.fillStyle = gradient;
    context.fillRect(0, 0, size, size);
    return canvas;
  }

  function somaPath(context, unit) {
    const points = 28;
    context.beginPath();
    for (let i = 0; i <= points; i++) {
      const theta = (i / points) * Math.PI * 2;
      const r = unit.radius * (1 + 0.24 * unit.b * Math.cos(3 * (theta + Math.PI / 2) + unit.phase * 0.3));
      const x = unit.x + Math.cos(theta) * r;
      const y = unit.y + Math.sin(theta) * r;
      if (i === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    }
    context.closePath();
  }

  /** Draws everything that doesn't move: connections, synapses, dendrites and cell bodies. */
  function drawStructure(context, network, colours) {
    const { units, edges } = network;
    context.lineCap = "round";
    context.lineJoin = "round";
    for (const edge of edges) {
      const { points } = edge.path;
      context.beginPath();
      context.moveTo(points[0], points[1]);
      for (let i = 2; i < points.length; i += 2) {
        context.lineTo(points[i], points[i + 1]);
      }
      context.strokeStyle = rgba(colours.line, colours.lineAlpha * (0.75 + 0.5 * edge.bend));
      context.lineWidth = 0.7 + 0.5 * edge.bend;
      context.stroke();
      if (edge.bend > 0.4) {
        const [x, y] = along(edge.path, edge.path.length - units[edge.to].radius - 3.5);
        context.beginPath();
        context.arc(x, y, 1.3, 0, Math.PI * 2);
        context.fillStyle = rgba(colours.soma, 0.55 * edge.bend);
        context.fill();
      }
    }
    for (const unit of units) {
      for (const [x1, y1, x2, y2, width] of unit.dendrites) {
        context.beginPath();
        context.moveTo(x1, y1);
        context.lineTo(x2, y2);
        context.strokeStyle = rgba(colours.soma, 0.6 * unit.b);
        context.lineWidth = width;
        context.stroke();
      }
    }
    for (const unit of units) {
      somaPath(context, unit);
      context.fillStyle = rgba(mix(colours.ground, colours.soma, 0.15 + 0.85 * unit.b), 1);
      context.fill();
      context.strokeStyle = rgba(colours.soma, 0.85);
      context.lineWidth = 1.2 - 0.4 * unit.b;
      context.stroke();
    }
  }

  /** Starts a network's activity: no spikes in flight, and the first volley soon. */
  function quiet() {
    return { clock: 0, nextVolley: 1.2, spikes: [] };
  }

  /** Fires a unit: it resets, flashes, and sends a spike down each of its connections. */
  function fire(state, unit) {
    unit.fired = state.clock;
    unit.potential = 0;
    unit.flash = 1;
    for (const edge of unit.out) {
      if (state.spikes.length < MAX_SPIKES) {
        state.spikes.push({ edge, distance: unit.radius });
      }
    }
  }

  /** Advances the activity by dt seconds: leak, spontaneous firing, volleys and arriving spikes. */
  function advance(network, state, dt) {
    state.clock += dt;
    const leak = Math.exp(-dt / LEAK);
    const fade = Math.exp(-dt / 0.24);
    for (const unit of network.units) {
      unit.potential *= leak;
      unit.flash *= fade;
      if (Math.random() < unit.rate * dt && state.clock - unit.fired > REFRACTORY) {
        fire(state, unit);
      }
    }
    if (state.clock >= state.nextVolley) {
      for (const id of network.layers[0]) {
        if (Math.random() < 0.6) {
          fire(state, network.units[id]);
        }
      }
      state.nextVolley = state.clock + VOLLEY_EVERY[0] + Math.random() * (VOLLEY_EVERY[1] - VOLLEY_EVERY[0]);
    }
    const arriving = [];
    state.spikes = state.spikes.filter((spike) => {
      spike.distance += SPEED * dt * (1.15 - 0.35 * spike.edge.bend);
      if (spike.distance >= spike.edge.path.length - network.units[spike.edge.to].radius) {
        arriving.push(spike.edge);
        return false;
      }
      return true;
    });
    for (const edge of arriving) {
      const target = network.units[edge.to];
      target.potential += edge.weight;
      if (target.potential >= THRESHOLD && state.clock - target.fired > REFRACTORY) {
        fire(state, target);
      }
    }
  }

  function start(root) {
    const structure = root.querySelector(".cg-hero__structure");
    const activity = root.querySelector(".cg-hero__activity");
    if (!structure || !activity) {
      return;
    }
    const still = window.matchMedia("(prefers-reduced-motion: reduce)");
    const toggle = root.parentElement.querySelector(".cg-hero__toggle");
    let paused = false;
    try {
      paused = localStorage.getItem("cg-hero-paused") === "1";
    } catch {
      paused = false;
    }
    let network = null;
    let colours = palette(root);
    let glow = sprite(colours.glow);
    let state = quiet();
    let frame = 0;
    let previous = 0;
    let visible = true;
    let width = 0;
    let height = 0;

    const drawActivity = () => {
      const context = activity.getContext("2d");
      context.clearRect(0, 0, width, height);
      context.globalCompositeOperation = colours.blend;
      for (const unit of network.units) {
        if (unit.flash < 0.02) {
          continue;
        }
        const size = (unit.radius * 5 + 10) * (0.7 + 0.5 * unit.flash);
        context.globalAlpha = unit.flash * 0.9;
        context.drawImage(glow, unit.x - size / 2, unit.y - size / 2, size, size);
        somaPath(context, unit);
        context.fillStyle = rgba(colours.spike, unit.flash);
        context.fill();
      }
      context.globalAlpha = 1;
      context.lineCap = "round";
      for (const spike of state.spikes) {
        const { path } = spike.edge;
        const [x, y] = along(path, spike.distance);
        const [tx, ty] = along(path, Math.max(0, spike.distance - TAIL));
        const gradient = context.createLinearGradient(tx, ty, x, y);
        gradient.addColorStop(0, rgba(colours.spike, 0));
        gradient.addColorStop(1, rgba(colours.spike, 0.85));
        context.beginPath();
        context.moveTo(tx, ty);
        const steps = 5;
        for (let i = 1; i <= steps; i++) {
          const [px, py] = along(path, Math.max(0, spike.distance - TAIL + (TAIL * i) / steps));
          context.lineTo(px, py);
        }
        context.strokeStyle = gradient;
        context.lineWidth = 1.8;
        context.stroke();
        context.drawImage(glow, x - 9, y - 9, 18, 18);
        context.beginPath();
        context.arc(x, y, 1.5, 0, Math.PI * 2);
        context.fillStyle = rgba(colours.spike, 1);
        context.fill();
      }
      context.globalCompositeOperation = "source-over";
    };

    const resize = (box) => {
      width = Math.round(box.width);
      height = Math.round(box.height);
      if (width < 10 || height < 10) {
        network = null;
        return;
      }
      const ratio = Math.min(2, window.devicePixelRatio || 1);
      for (const canvas of [structure, activity]) {
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);
        canvas.getContext("2d").setTransform(ratio, 0, 0, ratio, 0, 0);
      }
      network = build(width, height);
      state = quiet();
      redraw();
    };

    const redraw = () => {
      if (!network) {
        return;
      }
      const context = structure.getContext("2d");
      context.clearRect(0, 0, width, height);
      drawStructure(context, network, colours);
      if (still.matches) {
        settle();
      } else {
        drawActivity();
      }
    };

    // a still frame for reduced motion: run the network for a while, then draw once
    const settle = () => {
      for (let i = 0; i < 60 * 7; i++) {
        advance(network, state, 1 / 60);
      }
      drawActivity();
    };

    const tick = (time) => {
      frame = 0;
      if (!network || !visible || paused || document.hidden || still.matches) {
        previous = 0;
        return;
      }
      const dt = previous ? Math.min(0.05, (time - previous) / 1000) : 1 / 60;
      previous = time;
      advance(network, state, dt);
      drawActivity();
      frame = requestAnimationFrame(tick);
    };

    const play = () => {
      if (!frame && network && visible && !paused && !document.hidden && !still.matches) {
        frame = requestAnimationFrame(tick);
      }
    };

    // nothing moves under reduced motion, so the pause button only shows otherwise
    const label = () => {
      if (toggle) {
        toggle.hidden = still.matches;
        toggle.setAttribute("aria-pressed", String(paused));
      }
    };

    toggle?.addEventListener("click", () => {
      paused = !paused;
      try {
        localStorage.setItem("cg-hero-paused", paused ? "1" : "0");
      } catch {
        // the choice still holds for this visit
      }
      label();
      play();
    });

    new ResizeObserver((entries) => {
      resize(entries[0].contentRect);
      play();
    }).observe(root);

    new IntersectionObserver((entries) => {
      visible = entries[0].isIntersecting;
      play();
    }).observe(root);

    document.addEventListener("visibilitychange", play);

    still.addEventListener("change", () => {
      label();
      redraw();
      play();
    });

    label();

    new MutationObserver(() => {
      colours = palette(root);
      glow = sprite(colours.glow);
      redraw();
    }).observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
  }

  const begin = () => document.querySelectorAll(".cg-hero__art").forEach(start);
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", begin);
  } else {
    begin();
  }
})();
