/**
 * charts.js — Shared chart module for Simulation & Trading pages.
 *
 * Provides reusable Chart.js factories for:
 *   - Asset price curves with trade scatter overlay
 *   - Equity / capital line chart
 *   - Inspector lines (entry/exit/SL/TP)
 *   - Zoom/Pan synchronisation between paired charts
 *
 * Usage:
 *   const ctx = ChartModule.create({ ... });
 *   ctx.assetChart   — Chart.js instance (asset prices)
 *   ctx.equityChart  — Chart.js instance (equity/PnL curve)
 *   ctx.destroy()    — clean up both charts
 */

const ChartModule = (() => {

  /* ── constants ─────────────────────────────────────────── */
  const ASSET_COLORS = {
    GOLD:       '#f5c542',
    SILVER:     '#b0c4de',
    OIL_CRUDE:  '#e67e22',
    NATURALGAS: '#9b59b6',
  };

  const FONT = { size: 10, family: 'Share Tech Mono' };
  const GRID = { color: 'rgba(255,255,255,0.04)' };
  const TT   = { backgroundColor: '#18232e', borderColor: 'rgba(0,255,65,0.3)', borderWidth: 1, titleColor: '#c9d1d9', bodyColor: '#00ff41' };

  /* ── helpers ────────────────────────────────────────────── */
  function tsFmt(v) {
    const d = new Date(v);
    return d.toLocaleDateString('de-DE',{day:'2-digit',month:'2-digit'})
      + ' ' + d.toLocaleTimeString('de-DE',{hour:'2-digit',minute:'2-digit'});
  }

  function zoomRadius(chart, scaleId, data, minR, maxR) {
    if (!data.length) return minR;
    const scale = chart.scales[scaleId];
    if (!scale) return minR;
    const dataMin = data[0].x, dataMax = data[data.length - 1].x;
    const totalRange = dataMax - dataMin;
    if (totalRange <= 0) return maxR;
    const zf = totalRange / (scale.max - scale.min);
    if (zf >= 20) return maxR;
    if (zf >= 5)  return minR + (maxR - minR) * 0.7;
    if (zf >= 2)  return minR + (maxR - minR) * 0.3;
    return minR;
  }

  function getOrCreate(id) {
    const existing = Chart.getChart(id);
    if (existing) existing.destroy();
    return document.getElementById(id);
  }

  /* ── context object (one per page: sim / trading) ──────── */
  function create(opts) {
    /*
     * opts:
     *   assetCanvasId   — canvas element id for asset chart
     *   equityCanvasId  — canvas element id for equity chart
     *   resetBtnId      — reset-zoom button id (optional)
     *   onTradeSelect(idx)  — callback when trade dot clicked
     *   onTradeHover(idx)   — callback when trade dot hovered
     *   getSelectedIdx()    — returns currently selected tradeIdx or null
     *   getHoverIdx()       — returns currently hovered tradeIdx or null
     *   getTradeList()      — returns trade_list array
     *   isCapital           — bool, EUR capital mode vs point PnL
     *   equityLabel         — label for equity dataset (default: auto)
     */
    let assetChart = null;
    let equityChart = null;
    let _syncLock = false;
    let _detailDebounce = null;
    let _detailDbPath = null;

    const onSelect = opts.onTradeSelect || (() => {});
    const onHover  = opts.onTradeHover  || (() => {});
    const getSelIdx = opts.getSelectedIdx || (() => null);
    const getHovIdx = opts.getHoverIdx    || (() => null);
    const getTrades = opts.getTradeList   || (() => []);

    /* ── equity chart ──────────────────────────────────────── */
    function renderEquityChart(equityPoints, isCapital) {
      const canvas = getOrCreate(opts.equityCanvasId);
      if (!canvas) return;

      const yFmt = v => isCapital
        ? '€\u202f' + Number(v).toLocaleString('de-DE', {minimumFractionDigits:2, maximumFractionDigits:2})
        : Number(v).toFixed(4);

      equityChart = new Chart(canvas, {
        type: 'line',
        data: {
          datasets: [
            {
              label: opts.equityLabel || (isCapital ? 'Kapital' : 'Kum. PnL'),
              data: equityPoints,
              borderColor: '#3b82f6', borderWidth: 2,
              backgroundColor: 'rgba(59,130,246,0.07)', fill: true, tension: 0,
              pointRadius: ctx => {
                const idx = ctx.raw?.tradeIdx;
                if (idx == null) return 0;
                if (idx === getSelIdx()) return 7;
                if (idx === getHovIdx()) return 5;
                return zoomRadius(ctx.chart, 'x', ctx.dataset.data, 0, 4);
              },
              pointHoverRadius: 4,
              pointBackgroundColor: ctx => {
                const idx = ctx.raw?.tradeIdx;
                if (idx != null && idx === getSelIdx()) return '#f5c542';
                if (idx != null && idx === getHovIdx()) return '#fff';
                return '#3b82f6';
              },
              pointBorderColor: ctx => (ctx.raw?.tradeIdx != null && (ctx.raw.tradeIdx === getHovIdx() || ctx.raw.tradeIdx === getSelIdx())) ? '#fff' : '#3b82f6',
              pointBorderWidth: ctx => (ctx.raw?.tradeIdx != null && (ctx.raw.tradeIdx === getHovIdx() || ctx.raw.tradeIdx === getSelIdx())) ? 2 : 0,
              order: 10,
            },
            { _capInspect: 'entry', label: '_hidden', data: [], borderColor: 'rgba(255,255,255,0.5)', borderWidth: 1.5, borderDash: [], pointRadius: 0, fill: false, order: 5 },
            { _capInspect: 'exit',  label: '_hidden', data: [], borderColor: 'rgba(245,197,66,0.5)',  borderWidth: 1.5, borderDash: [], pointRadius: 0, fill: false, order: 5 },
          ]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          onClick: (event, elements) => {
            let idx = null;
            for (const el of elements) {
              const pt = equityChart.data.datasets[el.datasetIndex].data[el.index];
              if (pt && pt.tradeIdx != null) { idx = pt.tradeIdx; break; }
            }
            onSelect(idx);
          },
          plugins: {
            zoom: {
              zoom: {
                wheel: { enabled: true, speed: 0.08 }, pinch: { enabled: true }, mode: 'x',
                onZoomComplete: () => syncFromEquity(),
              },
              pan: { enabled: true, mode: 'x', onPanComplete: () => syncFromEquity() },
            },
            legend: { display: false },
            tooltip: { ...TT, callbacks: {
              title: items => tsFmt(items[0].parsed.x),
              label: item => item.dataset.label === '_hidden' ? null : `${item.dataset.label}: ${yFmt(item.parsed.y)}`,
            }}
          },
          scales: {
            x: { type: 'linear', display: false },
            y: { grid: GRID, ticks: { color: '#6e7681', font: FONT, callback: yFmt } }
          }
        }
      });
    }

    /* ── asset chart ───────────────────────────────────────── */
    function renderAssetChart() {
      const canvas = getOrCreate(opts.assetCanvasId);
      if (!canvas) return;

      assetChart = new Chart(canvas, {
        type: 'line',
        data: { datasets: [] },
        options: {
          responsive: true, maintainAspectRatio: false,
          onHover: (event, elements) => {
            if (!elements.length) { onHover(null); return; }
            for (const el of elements) {
              const pt = assetChart.data.datasets[el.datasetIndex].data[el.index];
              if (pt && pt.tradeIdx != null) { onHover(pt.tradeIdx); return; }
            }
          },
          onClick: (event, elements) => {
            let idx = null;
            for (const el of elements) {
              const pt = assetChart.data.datasets[el.datasetIndex].data[el.index];
              if (pt && pt.tradeIdx != null) { idx = pt.tradeIdx; break; }
            }
            onSelect(idx);
          },
          plugins: {
            zoom: {
              limits: { x: { minRange: 60 * 1000 } }, // min 1 minute visible
              zoom: {
                wheel: { enabled: true, speed: 0.08 }, pinch: { enabled: true }, mode: 'x',
                onZoomComplete: () => {
                  showResetBtn();
                  syncFromAsset();
                }
              },
              pan: { enabled: true, mode: 'x', onPanComplete: () => syncFromAsset() },
            },
            legend: {
              display: true,
              labels: { color: '#6e7681', font: FONT, boxWidth: 8, padding: 10,
                filter: item => item.text !== '_hidden' }
            },
            tooltip: { ...TT, callbacks: {
              title: items => tsFmt(items[0].parsed.x),
              label: item => {
                const raw = item.raw;
                const v   = typeof raw === 'object' ? raw.y : raw;
                const ds  = item.dataset.label;
                if (ds === '_hidden' && raw && raw.asset) {
                  const dir = raw.dir === 'BUY' ? '▲' : '▼';
                  return `${raw.asset} ${dir} ${Number(v).toFixed(2)} · Conf ${raw.conf}`;
                }
                if (ds === '_hidden') return null;
                return `${ds}: ${Number(v).toFixed(2)}`;
              }
            }}
          },
          scales: {
            x: { type: 'linear', grid: GRID, ticks: { color: '#6e7681', font: FONT, maxTicksLimit: 8, maxRotation: 0, callback: v => tsFmt(v) } },
            yR: { position: 'right', grid: GRID, ticks: { color: '#6e7681', font: FONT, callback: v => Number(v).toFixed(2) } }
          }
        }
      });
    }

    /* ── load asset price data + trade scatter ─────────────── */
    async function loadAssetCurves(apiUrl, assets, trades, startTs, endTs) {
      if (!apiUrl || !assets.length || !assetChart) return;
      let data;
      try {
        data = await apiFetch(
          apiUrl
          + `&assets=${encodeURIComponent(assets.join(','))}`
          + (startTs ? `&start=${encodeURIComponent(startTs)}` : '')
          + (endTs   ? `&end=${encodeURIComponent(endTs)}`     : '')
        );
      } catch (e) { console.warn('loadAssetCurves failed:', e); return; }

      const tradesByAsset = {};
      trades.forEach((t, i) => {
        if (!tradesByAsset[t.asset]) tradesByAsset[t.asset] = [];
        tradesByAsset[t.asset].push({ ...t, tradeIdx: i });
      });

      const newDatasets = [];
      for (const asset of assets) {
        const series = data[asset];
        if (!series || !series.timestamps.length) continue;
        const color = ASSET_COLORS[asset] || '#888';

        const priceData = series.timestamps.map((ts, i) => ({ x: new Date(ts).getTime(), y: series.prices[i] }));
        const assetTrades = tradesByAsset[asset] || [];
        if (assetTrades.length) {
          const existing = new Set(priceData.map(p => p.x));
          for (const t of assetTrades) {
            const xMs = new Date(t.entry_ts).getTime();
            if (!existing.has(xMs)) priceData.push({ x: xMs, y: t.entry_price });
          }
          priceData.sort((a, b) => a.x - b.x);
        }
        newDatasets.push({
          label: asset, yAxisID: 'yR', data: priceData,
          borderColor: color, borderWidth: 1.5,
          backgroundColor: 'transparent', fill: false, tension: 0,
          pointRadius: 0, pointHoverRadius: 3, order: 20,
        });

        if (assetTrades.length) {
          const scatterData = assetTrades.map(t => {
            const st = (t.status || '').toLowerCase();
            return {
              x: new Date(t.entry_ts).getTime(), y: t.entry_price,
              tradeIdx: t.tradeIdx, asset: t.asset, dir: t.direction,
              conf: t.confidence, status: t.status,
              isTP: st.includes('tp') || st.includes('profit'),
            };
          });
          newDatasets.push({
            label: '_hidden', yAxisID: 'yR', type: 'scatter', data: scatterData,
            pointRadius: ctx => {
              const idx = ctx.raw.tradeIdx;
              if (idx === getSelIdx()) return 7;
              if (idx === getHovIdx()) return 5;
              return zoomRadius(ctx.chart, 'x', ctx.dataset.data, 1.5, 5);
            },
            pointHoverRadius: 5,
            backgroundColor: ctx => {
              const r = ctx.raw;
              if (r.tradeIdx === getSelIdx()) return '#f5c542';
              if (r.tradeIdx === getHovIdx()) return '#fff';
              return r.isTP ? 'rgba(0,255,65,0.85)' : 'rgba(255,59,59,0.85)';
            },
            borderColor: ctx => {
              const r = ctx.raw;
              if (r.tradeIdx === getSelIdx()) return '#fff';
              return r.isTP ? 'rgba(0,255,65,0.5)' : 'rgba(255,59,59,0.5)';
            },
            borderWidth: ctx => ctx.raw.tradeIdx === getSelIdx() ? 2 : 1,
            order: 1,
          });
        }
      }

      if (!newDatasets.length) return;
      // Remove old asset curve + scatter datasets (keep inspector lines etc.)
      assetChart.data.datasets = assetChart.data.datasets.filter(
        ds => ds._inspectId || ds._detailId
      );
      assetChart.options.scales.yR.display = true;
      newDatasets.forEach(ds => assetChart.data.datasets.push(ds));
      assetChart.update('none');
    }

    /* ── inspector lines ──────────────────────────────────── */
    function updateInspectorLines() {
      if (!assetChart) return;
      const tl = getTrades();
      const selIdx = getSelIdx();
      const t = selIdx != null ? tl[selIdx] : null;

      const ids = ['_inspEntry', '_inspExit', '_inspTP', '_inspSL'];
      ids.forEach(id => {
        let ds = assetChart.data.datasets.find(d => d._inspectId === id);
        if (!ds) {
          ds = { _inspectId: id, label: '_hidden', yAxisID: 'yR', type: 'line',
                 data: [], pointRadius: 0, pointHoverRadius: 0, fill: false,
                 borderWidth: 1.5, order: 5 };
          assetChart.data.datasets.push(ds);
        }
        ds.data = [];
      });

      if (t) {
        const yMin = assetChart.scales.yR.min;
        const yMax = assetChart.scales.yR.max;
        const entryMs = new Date(t.entry_ts).getTime();
        const exitMs  = new Date(t.exit_ts || t.entry_ts).getTime();

        const dsEntry = assetChart.data.datasets.find(d => d._inspectId === '_inspEntry');
        const dsExit  = assetChart.data.datasets.find(d => d._inspectId === '_inspExit');
        const dsTP    = assetChart.data.datasets.find(d => d._inspectId === '_inspTP');
        const dsSL    = assetChart.data.datasets.find(d => d._inspectId === '_inspSL');

        dsEntry.data = [{ x: entryMs, y: yMin }, { x: entryMs, y: yMax }];
        dsEntry.borderColor = 'rgba(255,255,255,0.5)';
        dsExit.data  = [{ x: exitMs, y: yMin }, { x: exitMs, y: yMax }];
        dsExit.borderColor = 'rgba(245,197,66,0.5)';

        const ep = t.entry_price || 0;
        if (t.tp_price && Math.abs(t.tp_price - ep) / ep < 0.10) {
          dsTP.data = [{ x: entryMs, y: t.tp_price }, { x: exitMs, y: t.tp_price }];
          dsTP.borderColor = 'rgba(0,255,65,0.8)';
        }
        if (t.sl_price && Math.abs(t.sl_price - ep) / ep < 0.10) {
          dsSL.data = [{ x: entryMs, y: t.sl_price }, { x: exitMs, y: t.sl_price }];
          dsSL.borderColor = 'rgba(255,59,59,0.8)';
        }
      }
      assetChart.update('none');

      // equity inspector lines
      if (equityChart) {
        const dsCapEntry = equityChart.data.datasets.find(d => d._capInspect === 'entry');
        const dsCapExit  = equityChart.data.datasets.find(d => d._capInspect === 'exit');
        if (dsCapEntry) dsCapEntry.data = [];
        if (dsCapExit)  dsCapExit.data = [];
        if (t) {
          const capYMin = equityChart.scales.y.min;
          const capYMax = equityChart.scales.y.max;
          const entryMs = new Date(t.entry_ts).getTime();
          const exitMs  = new Date(t.exit_ts || t.entry_ts).getTime();
          if (dsCapEntry) dsCapEntry.data = [{ x: entryMs, y: capYMin }, { x: entryMs, y: capYMax }];
          if (dsCapExit)  dsCapExit.data  = [{ x: exitMs,  y: capYMin }, { x: exitMs,  y: capYMax }];
        }
        equityChart.update('none');
      }
    }

    /* ── zoom sync ────────────────────────────────────────── */
    function syncFromAsset() {
      if (_syncLock || !assetChart) return;
      _syncLock = true;
      const xMin = assetChart.scales.x?.min;
      const xMax = assetChart.scales.x?.max;
      if (equityChart && xMin != null && xMax != null) {
        equityChart.zoomScale('x', { min: xMin, max: xMax });
      }
      _syncLock = false;
      updateInspectorLines();
      debouncedDetailLoad(xMin, xMax);
    }

    function syncFromEquity() {
      if (_syncLock || !equityChart || !assetChart) return;
      _syncLock = true;
      const xMin = equityChart.scales.x?.min;
      const xMax = equityChart.scales.x?.max;
      if (xMin != null && xMax != null) {
        assetChart.zoomScale('x', { min: xMin, max: xMax });
        showResetBtn();
      }
      updateInspectorLines();
      debouncedDetailLoad(xMin, xMax);
      _syncLock = false;
    }

    function showResetBtn() {
      if (opts.resetBtnId) {
        const btn = document.getElementById(opts.resetBtnId);
        if (btn) btn.style.display = '';
      }
    }

    /* ── dynamic range loading (zoom/pan) ───────────────── */
    function setDetailDb(dbPath) { _detailDbPath = dbPath; }

    function debouncedDetailLoad(xMin, xMax) {
      if (_detailDebounce) clearTimeout(_detailDebounce);
      _detailDebounce = setTimeout(() => loadVisibleRange(xMin, xMax), 300);
    }

    /**
     * Fetches candles for the visible range from the DB.
     * Called on every zoom/pan — replaces dataset contents in-place.
     * Near live edge: extends range to now+1min for fresh data.
     */
    async function loadVisibleRange(fromMs, toMs) {
      if (!_detailDbPath || !assetChart || fromMs == null || toMs == null) return;

      // Near live edge: extend to now
      if ((Date.now() - toMs) < 5 * 60 * 1000) toMs = Date.now() + 60000;

      const assets = assetChart.data.datasets
        .filter(ds => ds.yAxisID === 'yR' && ds.label !== '_hidden' && !ds._inspectId)
        .map(ds => ds.label);
      if (!assets.length) return;

      const startIso = new Date(fromMs).toISOString();
      const endIso   = new Date(toMs).toISOString();
      try {
        const d = await apiFetch(
          `/api/sim/candles?db=${encodeURIComponent(_detailDbPath)}`
          + `&assets=${encodeURIComponent(assets.join(','))}`
          + `&start=${encodeURIComponent(startIso)}&end=${encodeURIComponent(endIso)}&max_points=2000`
        );
        const tl = getTrades();
        for (const asset of assets) {
          const series = d[asset];
          if (!series || !series.timestamps.length) continue;
          const ds = assetChart.data.datasets.find(dd => dd.label === asset && dd.yAxisID === 'yR');
          if (!ds) continue;
          const newData = series.timestamps.map((ts, i) => ({ x: new Date(ts).getTime(), y: series.prices[i] }));
          // Inject trade entry points
          const tradeSet = new Set(newData.map(p => p.x));
          for (const t of tl) {
            if (t.asset !== asset) continue;
            const eMs = new Date(t.entry_ts).getTime();
            if (!tradeSet.has(eMs) && eMs >= fromMs && eMs <= toMs) newData.push({ x: eMs, y: t.entry_price });
          }
          newData.sort((a, b) => a.x - b.x);
          // Replace in-place (preserves Chart.js reference)
          ds.data.length = 0;
          ds.data.push(...newData);
        }
        assetChart.update('none');
      } catch (e) { console.warn('loadVisibleRange failed:', e); }
    }

    function removeDetailCandles() {
      // No-op now — kept for API compatibility (resetZoom still calls it)
    }

    /* ── trade select helper (zoom to trade) ──────────────── */
    function zoomToTrade(tradeIdx) {
      const tl = getTrades();
      const t = tradeIdx != null ? tl[tradeIdx] : null;
      if (t && assetChart) {
        const entryMs = new Date(t.entry_ts).getTime();
        const exitMs  = new Date(t.exit_ts || t.entry_ts).getTime();
        const span    = Math.max(exitMs - entryMs, 60000);
        const pad     = span * 2;
        assetChart.zoomScale('x', { min: entryMs - pad, max: exitMs + pad });
        showResetBtn();
        syncFromAsset();
      }
      updateInspectorLines();
    }

    /* ── reset zoom ───────────────────────────────────────── */
    function resetZoom() {
      if (assetChart)  assetChart.resetZoom();
      if (equityChart) equityChart.resetZoom();
      if (opts.resetBtnId) {
        const btn = document.getElementById(opts.resetBtnId);
        if (btn) btn.style.display = 'none';
      }
      // Fire onReset callback so the page can reload its default range
      if (opts.onResetZoom) opts.onResetZoom();
    }

    /* ── live price append ────────────────────────────────── */
    /**
     * Append a live price point to the asset dataset.
     * If user is near the right edge, auto-scrolls to keep new data visible.
     */
    function appendLivePrice(asset, tsMs, price) {
      if (!assetChart) return;
      const ds = assetChart.data.datasets.find(
        dd => dd.label === asset && dd.yAxisID === 'yR' && !dd._inspectId
      );
      if (!ds) return;

      const data = ds.data;
      // Deduplicate
      if (data.length && data[data.length - 1].x >= tsMs) return;
      data.push({ x: tsMs, y: price });

      // Auto-scroll if zoomed and near the live edge
      const xScale = assetChart.scales.x;
      const isZoomed = assetChart.isZoomedOrPanned && assetChart.isZoomedOrPanned();
      if (isZoomed && xScale) {
        const margin = 3 * 60 * 1000;
        const prevLast = data.length >= 2 ? data[data.length - 2].x : tsMs;
        if (xScale.max >= prevLast - margin) {
          const visRange = xScale.max - xScale.min;
          assetChart.zoomScale('x', {
            min: tsMs - visRange + margin / 3,
            max: tsMs + margin / 3,
          });
          return; // zoomScale already calls update
        }
      }
      assetChart.update('none');
    }

    /* ── update (redraw without re-creating) ──────────────── */
    function update() {
      if (assetChart)  assetChart.update('none');
      if (equityChart) equityChart.update('none');
    }

    /* ── destroy ──────────────────────────────────────────── */
    function destroy() {
      if (assetChart)  { assetChart.destroy();  assetChart = null; }
      if (equityChart) { equityChart.destroy(); equityChart = null; }
    }

    return {
      renderAssetChart,
      renderEquityChart,
      loadAssetCurves,
      updateInspectorLines,
      zoomToTrade,
      resetZoom,
      update,
      destroy,
      setDetailDb,
      removeDetailCandles,
      appendLivePrice,
      get assetChart()  { return assetChart; },
      get equityChart() { return equityChart; },
    };
  }

  return { create, ASSET_COLORS, tsFmt, zoomRadius };
})();
