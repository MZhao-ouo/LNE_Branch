(function () {
  const payload = window.BRANCH_GLOBAL_SUVR_DATASETS;
  if (!payload || !payload.datasets) {
    return;
  }

  const svgNs = "http://www.w3.org/2000/svg";
  const windowMetricMeta = {
    global_suvr: { label: "Global SUVR" },
    age: { label: "Age" },
    PHC_MEM: { label: "PHC_MEM" },
    PHC_EXF: { label: "PHC_EXF" },
  };
  const state = {
    datasetKey: payload.datasetOrder[0],
    dataset: null,
    windowMetricKey: "global_suvr",
    windowMin: 0,
    windowMax: 0,
    selectedRowId: null,
    dragWindow: null,
  };

  const els = {
    datasetSelect: document.getElementById("dataset-select"),
    windowMetricSelect: document.getElementById("window-metric-select"),
    datasetDescription: document.getElementById("dataset-description"),
    dualRange: document.getElementById("dual-range"),
    minSlider: document.getElementById("window-min-slider"),
    maxSlider: document.getElementById("window-max-slider"),
    activeRange: document.getElementById("active-range"),
    rangeDrag: document.getElementById("range-drag"),
    windowMinValue: document.getElementById("window-min-value"),
    windowMaxValue: document.getElementById("window-max-value"),
    windowCenterValue: document.getElementById("window-center-value"),
    windowWidthValue: document.getElementById("window-width-value"),
    windowSpanLabel: document.getElementById("window-span-label"),
    pointsCount: document.getElementById("points-count"),
    pointsShare: document.getElementById("points-share"),
    datasetSize: document.getElementById("dataset-size"),
    datasetNote: document.getElementById("dataset-note"),
    branchCounts: document.getElementById("branch-counts"),
    diagnosisCounts: document.getElementById("diagnosis-counts"),
    plotLegend: document.getElementById("plot-legend"),
    plot: document.getElementById("trajectory-plot"),
    tableSummary: document.getElementById("table-summary"),
    tableBody: document.getElementById("points-table-body"),
    windowValueColumnLabel: document.getElementById("window-value-column-label"),
    selectionCard: document.getElementById("selection-card"),
  };

  function init() {
    buildDatasetSelect();
    bindEvents();
    loadDataset(state.datasetKey);
  }

  function buildDatasetSelect() {
    payload.datasetOrder.forEach((key) => {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = payload.datasets[key].label;
      els.datasetSelect.appendChild(option);
    });
  }

  function buildWindowMetricSelect() {
    els.windowMetricSelect.innerHTML = "";

    Object.keys(windowMetricMeta).forEach((key) => {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = windowMetricLabel(key);
      option.disabled = !state.dataset.window_metrics[key];
      if (option.disabled) {
        option.textContent += " (Unavailable)";
      }
      els.windowMetricSelect.appendChild(option);
    });
  }

  function bindEvents() {
    els.datasetSelect.addEventListener("change", () => {
      loadDataset(els.datasetSelect.value);
    });

    els.windowMetricSelect.addEventListener("change", () => {
      setActiveWindowMetric(els.windowMetricSelect.value);
    });

    els.minSlider.addEventListener("input", () => {
      setWindow(parseFloat(els.minSlider.value), state.windowMax);
    });

    els.maxSlider.addEventListener("input", () => {
      setWindow(state.windowMin, parseFloat(els.maxSlider.value));
    });

    els.rangeDrag.addEventListener("pointerdown", handleRangeDragStart);
    els.rangeDrag.addEventListener("pointermove", handleRangeDragMove);
    els.rangeDrag.addEventListener("pointerup", handleRangeDragEnd);
    els.rangeDrag.addEventListener("pointercancel", handleRangeDragEnd);

    els.tableBody.addEventListener("click", (event) => {
      const row = event.target.closest("tr[data-row-id]");
      if (!row) {
        return;
      }
      state.selectedRowId = Number(row.dataset.rowId);
      render();
    });
  }

  function loadDataset(key) {
    state.datasetKey = key;
    state.dataset = payload.datasets[key];
    state.selectedRowId = null;

    buildWindowMetricSelect();
    const nextMetricKey = state.dataset.window_metrics[state.windowMetricKey]
      ? state.windowMetricKey
      : state.dataset.window_metric_order[0];
    setActiveWindowMetric(nextMetricKey, true);
    els.datasetSelect.value = key;
    els.datasetDescription.textContent = "Source: " + state.dataset.source_csv;
    renderLegend();
    render();
  }

  function setActiveWindowMetric(key, skipRender) {
    const metricConfig = state.dataset.window_metrics[key];
    if (!metricConfig) {
      return;
    }

    state.windowMetricKey = key;
    [els.minSlider, els.maxSlider].forEach((slider) => {
      slider.min = metricConfig.min;
      slider.max = metricConfig.max;
      slider.step = metricConfig.step;
    });

    els.windowMetricSelect.value = key;
    setWindow(metricConfig.default_window.min, metricConfig.default_window.max, true);

    if (!skipRender) {
      render();
    }
  }

  function handleRangeDragStart(event) {
    if (event.button !== 0) {
      return;
    }

    event.preventDefault();
    state.dragWindow = {
      pointerId: event.pointerId,
      startClientX: event.clientX,
      startMin: state.windowMin,
      startMax: state.windowMax,
      sliderWidth: els.dualRange.getBoundingClientRect().width || 1,
    };
    els.rangeDrag.classList.add("is-dragging");
    els.rangeDrag.setPointerCapture(event.pointerId);
  }

  function handleRangeDragMove(event) {
    if (!state.dragWindow || state.dragWindow.pointerId !== event.pointerId) {
      return;
    }

    event.preventDefault();
    const range = currentWindowMetric();
    const totalSpan = range.max - range.min || 1;
    const deltaValue =
      ((event.clientX - state.dragWindow.startClientX) / state.dragWindow.sliderWidth) * totalSpan;
    const width = state.dragWindow.startMax - state.dragWindow.startMin;
    let nextMin = state.dragWindow.startMin + deltaValue;
    let nextMax = state.dragWindow.startMax + deltaValue;

    if (nextMin < range.min) {
      nextMax += range.min - nextMin;
      nextMin = range.min;
    }
    if (nextMax > range.max) {
      nextMin -= nextMax - range.max;
      nextMax = range.max;
    }

    nextMin = clamp(nextMin, range.min, range.max - width);
    nextMax = clamp(nextMin + width, range.min + width, range.max);
    setWindow(nextMin, nextMax);
  }

  function handleRangeDragEnd(event) {
    if (!state.dragWindow || state.dragWindow.pointerId !== event.pointerId) {
      return;
    }

    state.dragWindow = null;
    els.rangeDrag.classList.remove("is-dragging");
    if (els.rangeDrag.hasPointerCapture(event.pointerId)) {
      els.rangeDrag.releasePointerCapture(event.pointerId);
    }
  }

  function setWindow(nextMin, nextMax, skipRender) {
    const range = currentWindowMetric();
    let minValue = clamp(nextMin, range.min, range.max);
    let maxValue = clamp(nextMax, range.min, range.max);

    if (minValue > maxValue) {
      const swap = minValue;
      minValue = maxValue;
      maxValue = swap;
    }

    state.windowMin = roundToStep(minValue, currentWindowMetric().step);
    state.windowMax = roundToStep(maxValue, currentWindowMetric().step);
    els.minSlider.value = String(state.windowMin);
    els.maxSlider.value = String(state.windowMax);
    updateRangeFill();

    if (!skipRender) {
      render();
    }
  }

  function updateRangeFill() {
    const sliderMin = parseFloat(els.minSlider.min);
    const sliderMax = parseFloat(els.minSlider.max);
    const span = sliderMax - sliderMin || 1;
    const startPct = ((state.windowMin - sliderMin) / span) * 100;
    const endPct = ((state.windowMax - sliderMin) / span) * 100;
    els.dualRange.style.setProperty("--range-start", startPct + "%");
    els.dualRange.style.setProperty("--range-end", endPct + "%");
  }

  function render() {
    const filteredPoints = state.dataset.points
      .filter((point) => {
        const value = pointWindowValue(point);
        return value !== null && value >= state.windowMin && value <= state.windowMax;
      })
      .sort((left, right) => {
        return (
          pointWindowValue(left) - pointWindowValue(right) || left.row_id - right.row_id
        );
      });

    const visibleIds = new Set(filteredPoints.map((point) => point.row_id));
    if (state.selectedRowId !== null && !visibleIds.has(state.selectedRowId)) {
      state.selectedRowId = null;
    }

    const selectedPoint =
      state.selectedRowId === null
        ? null
        : filteredPoints.find((point) => point.row_id === state.selectedRowId) || null;

    renderMetrics(filteredPoints);
    renderPlot(filteredPoints, selectedPoint);
    renderTable(filteredPoints, selectedPoint);
    renderSelectionCard(selectedPoint, filteredPoints.length);
  }

  function renderMetrics(filteredPoints) {
    const totalPoints = state.dataset.points.length;
    const width = state.windowMax - state.windowMin;
    const center = state.windowMin + width / 2;
    const metricLabel = currentWindowMetricLabel();

    els.windowMinValue.textContent = formatMetricValue(state.windowMetricKey, state.windowMin);
    els.windowMaxValue.textContent = formatMetricValue(state.windowMetricKey, state.windowMax);
    els.windowCenterValue.textContent = formatMetricValue(state.windowMetricKey, center);
    els.windowWidthValue.textContent = formatMetricValue(state.windowMetricKey, width);
    els.windowSpanLabel.textContent =
      formatMetricValue(state.windowMetricKey, state.windowMin) +
      " to " +
      formatMetricValue(state.windowMetricKey, state.windowMax);

    els.pointsCount.textContent = String(filteredPoints.length);
    els.pointsShare.textContent =
      formatPercent(filteredPoints.length / totalPoints) +
      " of " +
      totalPoints +
      " records";

    els.datasetSize.textContent = String(totalPoints);
    els.datasetNote.textContent =
      state.dataset.unique_rids +
      " unique RIDs, " +
      state.dataset.suvr_column_count +
      " cortical SUVR columns";

    renderPills(
      els.branchCounts,
      state.dataset.branch_order.map((branch) => ({
        key: branch,
        label: state.dataset.branch_meta[branch].label,
        color: state.dataset.branch_meta[branch].point_color,
        count: filteredPoints.filter((point) => point.branch === branch).length,
      }))
    );

    renderPills(
      els.diagnosisCounts,
      state.dataset.diagnosis_order.map((diag) => ({
        key: String(diag),
        label: state.dataset.diagnosis_labels[String(diag)],
        color: diagnosisColor(diag),
        count: filteredPoints.filter((point) => point.lb === diag).length,
      }))
    );

    els.tableSummary.textContent =
      filteredPoints.length === 0
        ? "No records fall inside the current window."
        : filteredPoints.length +
          " records sorted by " +
          metricLabel +
          ". Click a row to focus the point.";
  }

  function renderPills(container, items) {
    container.innerHTML = "";
    items.forEach((item) => {
      const pill = document.createElement("div");
      pill.className = "pill";

      const dot = document.createElement("span");
      dot.className = "pill__dot";
      dot.style.background = item.color;

      const label = document.createElement("span");
      label.textContent = item.label + ": " + item.count;

      pill.appendChild(dot);
      pill.appendChild(label);
      container.appendChild(pill);
    });
  }

  function renderLegend() {
    els.plotLegend.innerHTML = "";

    const background = document.createElement("div");
    background.className = "legend-item";
    background.innerHTML =
      '<span class="legend-swatch" style="background: rgba(78, 68, 60, 0.35)"></span>Background cohort';
    els.plotLegend.appendChild(background);

    state.dataset.branch_order.forEach((branch) => {
      const item = document.createElement("div");
      item.className = "legend-item";
      item.innerHTML =
        '<span class="legend-swatch" style="background: ' +
        state.dataset.branch_meta[branch].point_color +
        '"></span>' +
        state.dataset.branch_meta[branch].label;
      els.plotLegend.appendChild(item);
    });
  }

  function renderPlot(filteredPoints, selectedPoint) {
    const dataset = state.dataset;
    const metricLabel = currentWindowMetricLabel();
    const width = 820;
    const height = 560;
    const margin = { top: 38, right: 24, bottom: 22, left: 24 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const xRange = dataset.extents.embedding1.max - dataset.extents.embedding1.min || 1;
    const yRange = dataset.extents.embedding2.max - dataset.extents.embedding2.min || 1;
    const xMin = dataset.extents.embedding1.min - xRange * 0.08;
    const xMax = dataset.extents.embedding1.max + xRange * 0.08;
    const yMin = dataset.extents.embedding2.min - yRange * 0.08;
    const yMax = dataset.extents.embedding2.max + yRange * 0.08;

    const xScale = (value) =>
      margin.left + ((value - xMin) / (xMax - xMin || 1)) * innerWidth;
    const yScale = (value) =>
      margin.top + innerHeight - ((value - yMin) / (yMax - yMin || 1)) * innerHeight;

    els.plot.innerHTML = "";
    els.plot.setAttribute(
      "aria-label",
      "Branch trajectory plot with a " +
        metricLabel +
        " window from " +
        formatMetricValue(state.windowMetricKey, state.windowMin) +
        " to " +
        formatMetricValue(state.windowMetricKey, state.windowMax) +
        "."
    );

    const frame = createSvg("rect", {
      x: 12,
      y: 12,
      width: width - 24,
      height: height - 24,
      rx: 18,
      class: "plot-border",
    });
    els.plot.appendChild(frame);

    const caption = createSvg("text", {
      x: 28,
      y: 36,
      class: "svg-caption",
    });
    caption.textContent =
      metricLabel +
      " window: " +
      formatMetricValue(state.windowMetricKey, state.windowMin) +
      " to " +
      formatMetricValue(state.windowMetricKey, state.windowMax);
    els.plot.appendChild(caption);

    const subcaption = createSvg("text", {
      x: 28,
      y: 56,
      class: "svg-subcaption",
    });
    subcaption.textContent =
      filteredPoints.length +
      " highlighted points across " +
      dataset.points.length +
      " total records";
    els.plot.appendChild(subcaption);

    dataset.branch_order.forEach((branch) => {
      const branchCenters = dataset.cluster_centers
        .filter((center) => center.branch === branch)
        .sort((left, right) => left.cluster - right.cluster);

      if (branchCenters.length < 2) {
        return;
      }

      const path = branchCenters
        .map((center, index) => {
          const x = xScale(center.embedding1);
          const y = yScale(center.embedding2);
          return (index === 0 ? "M " : "L ") + x + " " + y;
        })
        .join(" ");

      els.plot.appendChild(
        createSvg("path", {
          d: path,
          class: "trajectory-path",
          stroke: dataset.branch_meta[branch].line_color,
          "stroke-width": branch === "common_branch" ? 2.8 : 2.2,
        })
      );
    });

    const backgroundGroup = createSvg("g");
    dataset.points.forEach((point) => {
      backgroundGroup.appendChild(
        createSvg("circle", {
          cx: xScale(point.embedding1),
          cy: yScale(point.embedding2),
          r: 3.2,
          class: "point-bg",
        })
      );
    });
    els.plot.appendChild(backgroundGroup);

    const highlightGroup = createSvg("g");
    filteredPoints.forEach((point) => {
      const circle = createSvg("circle", {
        cx: xScale(point.embedding1),
        cy: yScale(point.embedding2),
        r: selectedPoint && selectedPoint.row_id === point.row_id ? 7.2 : 5.4,
        class:
          "point-highlight" +
          (selectedPoint && selectedPoint.row_id === point.row_id ? " point-selected" : ""),
        fill: dataset.branch_meta[point.branch].point_color,
        "data-row-id": point.row_id,
      });

      circle.addEventListener("click", () => {
        state.selectedRowId = point.row_id;
        render();
      });

      const title = createSvg("title");
      title.textContent =
        "RID " +
        point.RID +
        " | " +
        point.diag_label +
        " | " +
        point.branch_label +
        " | " +
        metricLabel +
        " " +
        formatMetricValue(state.windowMetricKey, pointWindowValue(point));
      circle.appendChild(title);
      highlightGroup.appendChild(circle);
    });
    els.plot.appendChild(highlightGroup);
  }

  function renderTable(filteredPoints, selectedPoint) {
    els.tableBody.innerHTML = "";
    els.windowValueColumnLabel.textContent = currentWindowMetricLabel();

    filteredPoints.forEach((point) => {
      const row = document.createElement("tr");
      row.dataset.rowId = String(point.row_id);
      if (selectedPoint && selectedPoint.row_id === point.row_id) {
        row.classList.add("is-selected");
      }

      row.innerHTML =
        "<td>" +
        point.RID +
        "</td>" +
        "<td>" +
        point.diag_label +
        "</td>" +
        "<td>" +
        point.branch_label +
        "</td>" +
        "<td>" +
        point.cluster +
        "</td>" +
        "<td>" +
        formatMetricValue(state.windowMetricKey, pointWindowValue(point)) +
        "</td>" +
        "<td>" +
        formatFloat(point.embedding1) +
        "</td>" +
        "<td>" +
        formatFloat(point.embedding2) +
        "</td>";
      els.tableBody.appendChild(row);
    });
  }

  function renderSelectionCard(selectedPoint, filteredCount) {
    if (!selectedPoint) {
      els.selectionCard.className = "selection-card is-empty";
      els.selectionCard.textContent =
        filteredCount === 0
          ? "No points fall inside the current window."
          : "Click a colored point or a table row to inspect one record.";
      return;
    }

    const cells = [
      selectionCell("RID", selectedPoint.RID),
      selectionCell("Diagnosis", selectedPoint.diag_label),
      selectionCell("Branch", selectedPoint.branch_label),
      selectionCell("Cluster", selectedPoint.cluster),
    ];

    metricSelectionOrder().forEach((metricKey) => {
      cells.push(
        selectionCell(
          windowMetricLabel(metricKey),
          formatMetricValue(metricKey, selectedPoint[metricKey])
        )
      );
    });

    cells.push(selectionCell("Embedding 1", formatFloat(selectedPoint.embedding1)));
    cells.push(selectionCell("Embedding 2", formatFloat(selectedPoint.embedding2)));
    cells.push(selectionCell("Row ID", selectedPoint.row_id));

    els.selectionCard.className = "selection-card";
    els.selectionCard.innerHTML = '<div class="selection-grid">' + cells.join("") + "</div>";
  }

  function selectionCell(label, value) {
    return "<div><span>" + label + "</span><strong>" + value + "</strong></div>";
  }

  function createSvg(tagName, attrs) {
    const node = document.createElementNS(svgNs, tagName);
    Object.entries(attrs || {}).forEach(([key, value]) => {
      node.setAttribute(key, value);
    });
    return node;
  }

  function clamp(value, minValue, maxValue) {
    return Math.min(Math.max(value, minValue), maxValue);
  }

  function roundToStep(value, step) {
    const scale = 1 / step;
    return Math.round(value * scale) / scale;
  }

  function formatFloat(value) {
    return Number(value).toFixed(3);
  }

  function currentWindowMetric() {
    return state.dataset.window_metrics[state.windowMetricKey];
  }

  function currentWindowMetricLabel() {
    return windowMetricLabel(state.windowMetricKey);
  }

  function windowMetricLabel(key) {
    const metricConfig = state.dataset && state.dataset.window_metrics[key];
    if (metricConfig) {
      return metricConfig.label;
    }
    return windowMetricMeta[key].label;
  }

  function pointWindowValue(point) {
    const value = point[state.windowMetricKey];
    return value === null || value === undefined ? null : Number(value);
  }

  function metricSelectionOrder() {
    const keys = [state.windowMetricKey];
    ["global_suvr", "age", "PHC_MEM", "PHC_EXF"].forEach((key) => {
      if (keys.includes(key) || !state.dataset.window_metrics[key]) {
        return;
      }
      keys.push(key);
    });
    return keys;
  }

  function formatMetricValue(metricKey, value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "NA";
    }
    return Number(value).toFixed(metricDigits(metricKey));
  }

  function metricDigits(metricKey) {
    if (metricKey === "age") {
      return 2;
    }
    return 3;
  }

  function formatPercent(value) {
    return (value * 100).toFixed(1) + "%";
  }

  function diagnosisColor(diag) {
    if (diag === 1) {
      return "#2A9D8F";
    }
    if (diag === 2) {
      return "#E09F3E";
    }
    return "#C44536";
  }

  init();
})();
