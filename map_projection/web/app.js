"use strict";
const $ = id => document.getElementById(id);
let state, initialized = false, lastFrame = -1, selectedCorner = 0;
let repeatTimer, repeatDelay, held = false, nudgeBusy = false, pollBusy = false;
let mutations=Promise.resolve(), lastPreviewSize="", lastServerError="";
let dateMode="today", draggingTime=false, scrubUntil=0;
let livePatch=null, livePersist=false, liveBusy=false;
let previewBusy=false, previewTimer, previewAt=0;
const speedSeconds=()=>Math.pow(3600,value("sun-speed")/1000);
const value = id => Number($(id).value);
const showError = error => { $("error").textContent = String(error); $("error").hidden = !error; };

async function request(path, data) {
  const response = await fetch(path, data === undefined ? {cache:"no-store"} : {
    method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(data)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  update(result);
  return result;
}
function api(path,data) {
  if(data===undefined)return request(path);
  if(path!=="/api/nudge")stopRepeat();
  // Keep a late jog response from switching back after a sunlight request.
  const task=mutations.catch(()=>{}).then(async()=>{
    if(livePatch)flushLive();
    const end=performance.now()+1500;
    while((livePatch||liveAck<liveSequence)&&performance.now()<end)await new Promise(resolve=>setTimeout(resolve,10));
    return request(path,data);
  });
  mutations=task;
  return task;
}
function refreshPreview() {
  // A single independent 15 Hz MJPEG stream. No image request per command,
  // render event, acknowledgement, or image-load callback.
  if(document.hidden || !state || !state.frame_id)return;
  if($("preview").dataset.streaming!=="1"){
    $("preview").dataset.streaming="1";
    $("preview").src=`/preview.mjpg?session=${Date.now()}`;
  }
  $("preview").parentElement.style.aspectRatio=String(state.config.output.width/state.config.output.height);
}
$("preview").addEventListener("error",()=>{
  $("preview").dataset.streaming="0";
  clearTimeout(previewTimer);previewTimer=setTimeout(refreshPreview,500);
});
function action(id, fn) {
  $(id).addEventListener("click", async () => {
    $(id).disabled = true;
    try { showError(""); await fn(); } catch (error) { showError(error.message); }
    finally { $(id).disabled = false; }
  });
}
function clock(minutes) {
  return `${String(Math.floor(minutes/60)).padStart(2,"0")}:${String(minutes%60).padStart(2,"0")}`;
}
function fillTime(time,setDate=true) {
  if(setDate)$("sun-date").value = time.slice(0,10);
  const minute = Number(time.slice(11,13))*60 + Number(time.slice(14,16));
  $("sun-minute").value = minute;
  $("clock").textContent = clock(minute);
  timelineProgress();
}
function fill(c) {
  dateMode=c.sun.date_mode;
  fillTime(c.sun.time);
  const fields = {
    "sun-step":c.sun.step_minutes, "shadow-size":c.sun.shadow_size, ambient:c.sun.ambient,
    "sun-speed":1000*Math.log(c.sun.minutes_per_second*60)/Math.log(3600),"sun-backend":c.sun.backend,
    "sun-renderer":c.sun.renderer,"sun-terrain":c.sun.terrain,"grid-width":c.sun.grid_width,
    "sun-lat":c.sun.latitude,"sun-lon":c.sun.longitude,"shadow-bias":c.sun.bias_mm,
    "pose-x":c.projector.x,"pose-y":c.projector.y,"pose-z":c.projector.z,"roll":c.projector.roll,
    "throw-ratio":c.projector.throw_ratio,"shift-x":c.projector.shift_x,"shift-y":c.projector.shift_y,
    width:c.output.width,height:c.output.height,refresh:c.output.refresh,fullscreen:String(c.output.fullscreen),mirror:String(c.output.mirror),
    "pixel-x":c.pixel.x,"pixel-y":c.pixel.y,pattern:c.pattern,
    "geo-lat-min":c.geo.lat_min,"geo-lat-max":c.geo.lat_max,"geo-lon-min":c.geo.lon_min,"geo-lon-max":c.geo.lon_max,
    "geo-width":c.geo.width_mm,"geo-height":c.geo.height_mm
  };
  for (const [id,v] of Object.entries(fields)) $(id).value = v;
  $("chromatic-strength").value=c.optics.chromatic_pixels;chromaticReadout();
  speedReadout();
  $("installation-zone").textContent=c.sun.timezone;
}
function update(s) {
  if(state && s.instance===state.instance && (s.revision<state.revision || (s.revision===state.revision && s.frame_id<state.frame_id)))return;
  if(state && s.instance!==state.instance){initialized=false;lastFrame=-1;previewBusy=false;}
  state = s;
  if (!initialized) { fill(s.config); initialized = true; }
  $("connection").textContent = "Local server connected";
  $("connection-dot").classList.add("online");
  const serverError=s.error || s.output.error || "";
  if(serverError)showError(serverError);
  else if(lastServerError && $("error").textContent===lastServerError)showError("");
  lastServerError=serverError;
  $("resolution").textContent = `${s.config.output.width} × ${s.config.output.height}`;
  $("render-status").textContent = s.pending && s.config.pattern==="sun" ?
    `${s.status} · showing ${s.displayed_pattern||"previous frame"}` : s.status;
  const seconds=s.timings.backend?.startsWith("GPU") ? s.timings.display_frame_seconds : s.timings.frame_seconds;
  const measuringRate=(s.config.sun.playing || draggingTime || s.motion || s.track.animating || s.layers?.rivers?.animating) && s.timings.display_fps!==undefined;
  $("render-time-label").textContent=measuringRate?"OUTPUT UPDATES":s.timings.backend?.startsWith("GPU")?"LAST DRAW & SWAP":"LAST COMPUTATION";
  $("render-time").textContent = measuringRate ? `${s.timings.display_fps.toFixed(1)} fps` : seconds !== undefined ? `${(seconds*1000).toFixed(1)} ms` : "—";
  $("altitude").textContent = `${s.sun.altitude.toFixed(1)}°`;
  playbackState(s.config.sun.playing,s.motion);
  const enabled=s.config.pattern==="sun" && s.config.sun.enabled;
  $("shadow-toggle").textContent=enabled?"Disable Shadows":"Enable Shadows";
  $("shadow-toggle").setAttribute("aria-pressed",String(enabled));
  const shown=s.displayed_time||s.config.sun.time;
  if(!draggingTime && document.activeElement!==$("sun-date")){
    $("sun-date").value=s.config.sun.time.slice(0,10);
    if(canFollow(shown))fillTime(shown,false);
  }
  cacheState(s.date_cache);
  updateLayers(s);
  $("track-toggle").textContent=s.config.track.enabled?"Hide track":"Show track";
  $("track-toggle").setAttribute('aria-pressed',String(s.config.track.enabled));
  if(document.activeElement!==$("track-color"))$("track-color").value=s.track.color;
  $("track-status").textContent=s.track.error || (s.track.ready?`${s.track.points.toLocaleString()} points · ${s.track.distance_km.toFixed(1)} km`:s.config.track.enabled?"Registering track on the printed surface…":"Ready to reveal on the map.");
  $("track-replay").disabled=!s.config.track.enabled;
  $("loading").hidden = s.frame_id > 0;
  refreshPreview();
  const [x,y] = s.corners[selectedCorner];
  $("corner-position").textContent = `Corner ${selectedCorner+1} · X ${x.toFixed(1)} / Y ${y.toFixed(1)} px`;
  const output = s.output;
  $("output-label").textContent = output.direct_scanout ? "Direct KMS/DRM output" : output.pixel_buffer_verified ? "Native-size desktop output" : "Output verification";
  $("output-note").textContent = output.error || output.note || "Output is starting.";
  $("display-diagnostics").textContent = JSON.stringify(output,null,2);
  if (output.displays && $("display").dataset.loaded !== JSON.stringify(output.displays.map(d=>d.name))) {
    $("display").replaceChildren(...output.displays.map(d => new Option(`${d.index} · ${d.name}`,String(d.index))));
    $("display").value = s.config.output.display;
    $("display").dataset.loaded = JSON.stringify(output.displays.map(d=>d.name));
  }
  $("mesh-summary").textContent = `${s.mesh.name} · ${(s.mesh.triangles/1e6).toFixed(2)} M triangles · ${s.mesh.width_mm.toFixed(1)} × ${s.mesh.height_mm.toFixed(1)} mm`;
  $("cache-summary").textContent = `Cache ${(s.cache_bytes/1048576).toFixed(0)} MB`;
  $("cancel-bake").hidden = !s.bake.remaining;
  $("bake").disabled = !!s.bake.remaining || ["reference","precise","events"].includes(s.config.sun.backend);
  $("bake-status").textContent = s.config.sun.backend==="events" ? "Daily horizon events · generated on this device. Daily caches use up to 10 GB." : ["reference","precise"].includes(s.config.sun.backend) ? "Precise DEM rays · live rendering, no cached approximations." : s.bake.remaining ? `Caching ${s.bake.total-s.bake.remaining} / ${s.bake.total} frames. Controls remain available.` : s.config.sun.renderer==="live" ? "Live mode recalculates arbitrary sun positions on this device. Caching is optional." : "Reference mode reads the mesh per sun position. Cached playback is faster.";
  if (s.query) {
    const q=s.query;
    const lines=[q.status];
    if (q.pixel) lines.push(`Pixel: ${q.pixel.join(", ")}`);
    if (q.map_mm) lines.push(`Map mm: ${q.map_mm.map(v=>v.toFixed(3)).join(", ")}`);
    if (q.latitude !== undefined) lines.push(`WGS84: ${q.latitude.toFixed(6)}, ${q.longitude.toFixed(6)}`);
    if (q.pixel_error_mm !== undefined && q.pixel_error_mm !== null) lines.push(`Pixel-center error: ${q.pixel_error_mm.toFixed(3)} mm`);
    if (q.pixel_hit_mm) lines.push(`Actual pixel hit mm: ${q.pixel_hit_mm.map(v=>v.toFixed(3)).join(", ")}`);
    $("query-result").textContent=lines.join("\n");
  }
}
function tab(name) {
  document.querySelectorAll(".tab").forEach(button => {
    const active=button.dataset.tab===name;
    button.classList.toggle("active",active); button.setAttribute("aria-selected",String(active));
  });
  document.querySelectorAll(".tab-panel").forEach(panel => panel.hidden=panel.id!==`tab-${name}`);
}
document.querySelectorAll(".tab").forEach(button=>button.addEventListener("click",()=>tab(button.dataset.tab)));
function settingsPatch() {
  return {sun:{renderer:$("sun-renderer").value,terrain:$("sun-terrain").value,grid_width:value("grid-width"),
    step_minutes:value("sun-step"),shadow_size:value("shadow-size"),ambient:value("ambient"),bias_mm:value("shadow-bias"),
    backend:$("sun-backend").value,latitude:value("sun-lat"),longitude:value("sun-lon")}};
}
function timelineProgress() {
  $("timeline").style.setProperty("--progress",value("sun-minute")/1439);
  $("sun-minute").style.setProperty("--fill",`${100*value("sun-minute")/1439}%`);
  $("sun-minute").setAttribute("aria-valuetext",clock(value("sun-minute")));
}
function speedReadout() {
  const seconds=speedSeconds();
  const format=n=>n<10?Number(n.toFixed(1)):Math.round(n);
  const label=seconds>=3599?"1 h/s":seconds>=60?`${format(seconds/60)} min/s`:`${format(seconds)} s/s`;
  $("speed-readout").textContent=label;$("sun-speed").setAttribute("aria-valuetext",label);
  $("sun-speed").style.setProperty("--fill",`${value("sun-speed")/10}%`);
}
function playbackState(playing,motion) {
  const label=playing?"Stop Animation":"Start Animation";
  if($("play").textContent!==label)$("play").textContent=label;
  $("play").setAttribute("aria-pressed",String(playing));
  $("timeline").classList.toggle("animated",playing||!!motion);
  $("time-motion").textContent=playing?"▶":motion==="return"?"↩":motion?"↗":"";
  const status=playing?"Animating":motion==="return"?"Returning to selection":motion?"Moving to now":"Manual";
  if($("playback-status").textContent!==status)$("playback-status").textContent=status;
  $("playback-status").classList.toggle("active",playing||!!motion);
}
// Ordered, bidirectional controls: slider input never waits for a response or
// preview. Keep one latest unsent patch; TCP backpressure cannot build a backlog.
let liveSocket,liveSequence=0,liveRaf=0,liveAck=0,pendingScrub=null;
function mergePatch(a,b){
  const result={...a};
  for(const [key,v] of Object.entries(b))result[key]=v&&typeof v==="object"&&!Array.isArray(v)?mergePatch(result[key]||{},v):v;
  return result;
}
function connectLive(){
  liveSocket=new WebSocket(`${location.protocol==="https:"?"wss":"ws"}://${location.host}/api/live`);
  liveSocket.onopen=()=>{if(livePatch)scheduleLive();};
  liveSocket.onmessage=event=>{
    const msg=JSON.parse(event.data);
    if(msg.type==="error")showError(msg.error);
    if(msg.type==="ack"||msg.type==="error")liveAck=Math.max(liveAck,msg.seq||0);
  };
  liveSocket.onclose=()=>setTimeout(connectLive,500);
}
function scheduleLive(){if(!liveRaf)liveRaf=requestAnimationFrame(flushLive);}
function liveUpdate(patch,persist=false){
  livePatch=mergePatch(livePatch||{},patch);livePersist ||= persist;scheduleLive();
}
function flushLive(){
  liveRaf=0;
  if(!livePatch)return;
  if(liveSocket?.readyState!==WebSocket.OPEN || liveSocket.bufferedAmount>8192){setTimeout(scheduleLive,16);return;}
  const patch=livePatch,persist=livePersist;livePatch=null;livePersist=false;
  liveSocket.send(JSON.stringify({type:"config",seq:++liveSequence,patch,persist}));
}
connectLive();
setInterval(()=>{if(liveSocket?.readyState===WebSocket.OPEN)liveSocket.send(JSON.stringify({type:"ping"}));},15000);
function canFollow(time){
  if(draggingTime)return false;
  if(pendingScrub && performance.now()<scrubUntil && time.slice(11,16)!==pendingScrub)return false;
  pendingScrub=null;return true;
}
function cacheState(cache){
  if(!cache)return;
  $("sun-date").classList.toggle("pending-date",cache.pending);
  $("cache-used-date").hidden=!cache.pending;
  $("cache-used-date").textContent=cache.active_date?`Showing ${cache.active_date}`:"Preparing first day";
  const job=cache.generation||{},busy=job.state==="generating";
  $("day-generation").hidden=!busy&&!cache.pending;
  if(busy)$("day-progress").value=job.progress||0;else $("day-progress").removeAttribute("value");
  $("day-generation-label").textContent=job.state==="error"?job.error:busy?`${job.stage} · ${job.date} · ${Math.floor((job.progress||0)*100)}%`:`Queued · ${cache.requested_date}`;
}
function chromaticReadout(){
 const n=value("chromatic-strength");$("chromatic-readout").textContent=n===0?"Off":`${n>0?"+":""}${n.toFixed(1)} px`;
}
function scrub(persist=false) {
  if(!$("sun-date").value)return;
  scrubUntil=performance.now()+1500;pendingScrub=clock(value("sun-minute"));
  $("clock").textContent=clock(value("sun-minute"));timelineProgress();playbackState(false,null);
  liveUpdate({pattern:"sun",sun:{time:`${$("sun-date").value}T${clock(value("sun-minute"))}:00`,date_mode:dateMode,playing:false}},persist);
}
$("sun-minute").addEventListener("pointerdown",()=>{draggingTime=true;});
for(const name of ["pointerup","pointercancel"])window.addEventListener(name,()=>{draggingTime=false;});
$("sun-minute").addEventListener("input",()=>scrub());
$("sun-minute").addEventListener("change",()=>scrub(true));
$("sun-speed").addEventListener("input",()=>{speedReadout();liveUpdate({sun:{minutes_per_second:speedSeconds()/60}});});
$("sun-speed").addEventListener("change",()=>liveUpdate({sun:{minutes_per_second:speedSeconds()/60}},true));
action("play",()=>api(state.config.sun.playing?"/api/stop":"/api/play",{}));
action("sun-now",()=>api("/api/now",{}));
action("shadow-toggle",()=>api("/api/config",{pattern:"sun",sun:{enabled:!(state.config.pattern==="sun" && state.config.sun.enabled)}}));
$("sun-date").addEventListener("change",()=>{dateMode="fixed";scrub(true);});
action("blackout",()=>api("/api/config",{pattern:"black",sun:{playing:false}}));
action("show-calibration",()=>api("/api/config",{pattern:"calibration",sun:{playing:false}}));
action("reset-corners",()=>api("/api/config",{corners:null,pattern:"calibration",sun:{playing:false}}));
action("pose-apply",()=>api("/api/config",{projector:{x:value("pose-x"),y:value("pose-y"),z:value("pose-z"),roll:value("roll"),throw_ratio:value("throw-ratio"),shift_x:value("shift-x"),shift_y:value("shift-y")}}));
action("output-apply",()=>api("/api/config",{output:{width:value("width"),height:value("height"),display:value("display"),refresh:value("refresh"),fullscreen:$("fullscreen").value==="true",mirror:$("mirror").value==="true"},...(state.config.output.width!==value("width")||state.config.output.height!==value("height")?{corners:null}:{})}));
$("sun-terrain").addEventListener("change",()=>{if($("sun-terrain").value==="dem")$("sun-renderer").value="live";});
$("sun-backend").addEventListener("change",()=>{
  if(["gpu","reference","precise","events"].includes($("sun-backend").value)){$("sun-terrain").value="dem";$("sun-renderer").value="live";}
});
action("pattern-apply",()=>api("/api/config",{pattern:$("pattern").value,pixel:{x:value("pixel-x"),y:value("pixel-y")},sun:{playing:false}}));
action("target-geo",()=>api("/api/query",{kind:"geo",latitude:value("target-lat"),longitude:value("target-lon")}));
action("target-map",()=>api("/api/query",{kind:"map",x:value("target-x"),y:value("target-y")}));
action("geo-apply",()=>api("/api/config",{geo:{lat_min:value("geo-lat-min"),lat_max:value("geo-lat-max"),lon_min:value("geo-lon-min"),lon_max:value("geo-lon-max"),width_mm:value("geo-width"),height_mm:value("geo-height")}}));
for(const id of ["sun-renderer","sun-terrain","grid-width","ambient","sun-backend","sun-lat","sun-lon","shadow-size","shadow-bias","sun-step"])
  $(id).addEventListener("change",()=>api("/api/config",settingsPatch()).catch(e=>showError(e.message)));
action("bake",async()=>{await api("/api/config",{sun:{playing:false}});await api("/api/bake",{});});
action("cancel-bake",()=>api("/api/cancel",{}));
document.querySelectorAll("[data-corner]").forEach(button=>button.addEventListener("click",()=>{
  selectedCorner=Number(button.dataset.corner);
  document.querySelectorAll("[data-corner]").forEach(b=>b.classList.toggle("selected",b===button));
  if(state)update(state);
}));
function stopRepeat() { held=false;clearInterval(repeatTimer);clearTimeout(repeatDelay); }
async function nudge(dx,dy) {
  if(nudgeBusy || !held)return;
  nudgeBusy=true;
  try{await api("/api/nudge",{corner:selectedCorner,dx,dy,step:value("nudge-step")});}catch(error){showError(error.message);stopRepeat();}
  finally{nudgeBusy=false;}
}
document.querySelectorAll("[data-move]").forEach(button=>{
  button.addEventListener("pointerdown",event=>{
    if(event.button!==0)return;
    event.preventDefault();stopRepeat();held=true;button.setPointerCapture(event.pointerId);
    const [dx,dy]=button.dataset.move.split(",").map(Number);nudge(dx,dy);
    repeatDelay=setTimeout(()=>{repeatTimer=setInterval(()=>nudge(dx,dy),70);},300);
  });
  ["pointerup","pointercancel","lostpointercapture"].forEach(name=>button.addEventListener(name,stopRepeat));
  button.addEventListener("keydown",event=>{
    if((event.key==="Enter"||event.key===" ")&&!event.repeat){event.preventDefault();held=true;const [dx,dy]=button.dataset.move.split(",").map(Number);nudge(dx,dy).finally(stopRepeat);}
  });
});
window.addEventListener("blur",stopRepeat);
$("nudge-step").addEventListener("change",()=>{
  stopRepeat(); $("nudge-step-label").textContent=`${value("nudge-step")} px`;
});
let resizeTimer;
new ResizeObserver(()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(refreshPreview,150);}).observe($("preview").parentElement);
document.addEventListener("visibilitychange",()=>{
  if(document.hidden){stopRepeat();$("preview").dataset.streaming="0";$("preview").removeAttribute("src");}
  else refreshPreview();
});
$("preview").addEventListener("click",event=>{
  if(!state)return;
  const rect=event.currentTarget.getBoundingClientRect();
  const x=Math.min(state.config.output.width-1,Math.floor((event.clientX-rect.left)/rect.width*state.config.output.width));
  const y=Math.min(state.config.output.height-1,Math.floor((event.clientY-rect.top)/rect.height*state.config.output.height));
  tab("target");api("/api/query",{kind:"pixel",x,y}).catch(e=>showError(e.message));
});
async function poll() {
  if(pollBusy)return;pollBusy=true;
  try{await api("/api/state");}catch(error){$("connection").textContent="Server disconnected";$("connection-dot").classList.remove("online");}
  finally{pollBusy=false;}
}
poll();setInterval(poll,1000);

const events=new EventSource("/api/events");
events.onmessage=event=>{
  const tick=JSON.parse(event.data);
  if(!state || tick.instance!==state.instance){poll();return;}
  if(tick.revision<state.revision)return;
  state.frame_id=tick.frame_id;
  state.config.sun.playing=tick.playing;state.motion=tick.motion;
  playbackState(tick.playing,tick.motion);
  if(tick.time && canFollow(tick.time))fillTime(tick.time,false);
  if(Number.isFinite(tick.display_fps) && (draggingTime||tick.playing||tick.motion)){$("render-time-label").textContent="OUTPUT UPDATES";$("render-time").textContent=`${tick.display_fps.toFixed(1)} fps`;}
  refreshPreview();
};

action("track-toggle",()=>api("/api/config",{pattern:"sun",track:{enabled:!state.config.track.enabled}}));
action("track-replay",()=>api("/api/track/replay",{}));
action("track-random",async()=>{await api("/api/config",{track:{color:null}});await api("/api/track/replay",{});});
$("track-color").addEventListener("input",()=>liveUpdate({track:{color:$("track-color").value}}));
$("track-color").addEventListener("change",()=>liveUpdate({track:{color:$("track-color").value}},true));
$("chromatic-strength").addEventListener("input",()=>{chromaticReadout();liveUpdate({optics:{chromatic_pixels:value("chromatic-strength")}});});
$("chromatic-strength").addEventListener("change",()=>liveUpdate({optics:{chromatic_pixels:value("chromatic-strength")}},true));
action("chromatic-pattern",()=>api("/api/config",{pattern:"chromatic",sun:{playing:false}}));
action("chromatic-map",()=>api("/api/config",{pattern:"sun"}));

// All geographic backgrounds and overlays share the same layer overview.
function layerReadout(name,key){
 const input=$(`layer-${name}-${key}`),output=$(`layer-${name}-${key}-value`);if(!output)return;
 const n=Number(input.value);output.textContent=key==='brightness'?`${n.toFixed(2)}×`:key==='opacity'?`${Math.round(n*100)}%`:(key==='width'||key==='river_thickness')?`${n} px`:`${n.toLocaleString()} m`;
 input.style.setProperty('--fill',`${100*(n-Number(input.min))/(Number(input.max)-Number(input.min))}%`);
}
function updateLayers(s){
 for(const [name,layer] of Object.entries(s.config.layers)){
  for(const [key,v] of Object.entries(layer)){
   const input=$(`layer-${name}-${key}`);if(!input)continue;
   if(document.activeElement!==input){if(input.type==='checkbox')input.checked=v;else input.value=String(v);}
   layerReadout(name,key);
  }
 }
 $('layer-contours-level').closest('label').hidden=!s.config.layers.contours.single;
 $('layer-contours-fill').closest('label').hidden=!s.config.layers.contours.single;
 $('layer-contours-width').closest('label').hidden=s.config.layers.contours.single&&s.config.layers.contours.fill!=='line';
 $('layer-contours-interval').closest('label').hidden=s.config.layers.contours.single;
 $('layer-track-enabled').checked=s.config.track.enabled;
 $('layer-shadow-enabled').checked=s.config.sun.enabled;
 $('shadow-floor-enabled').checked=s.config.sun.shadow_floor;
 $('shadow-normalize').checked=s.config.sun.normalize;
 if(document.activeElement!==$('shadow-floor-amount'))$('shadow-floor-amount').value=s.config.sun.shadow_floor_amount;
 $('shadow-floor-value').textContent=`${Math.round(s.config.sun.shadow_floor_amount*100)}%`;
 $('shadow-floor-amount').disabled=!s.config.sun.shadow_floor;
 if(document.activeElement!==$('track-width'))$('track-width').value=s.config.track.width;
 $('track-width-value').textContent=`${$('track-width').value} px`;
 const data=s.layers||{};
 $('rivers-replay').disabled=!s.config.layers.rivers.enabled;
 $('divide-count').textContent=data.divides?.basins?`${data.divides.basins.toLocaleString()} drainage basins · HydroBASINS level ${data.divides.level}. Coastlines excluded.`:'Boundaries between drainage basins.';
 $('layers-status').textContent=data.error||(!data.ready?'Map data are being prepared for this installation.':'');
 $('show-layers').hidden=s.config.pattern==='sun';
 $('river-count').textContent=data.rivers?.reaches?`${data.rivers.reaches.toLocaleString()} river reaches. Includes catchments ≥10 km² or mean flow ≥0.1 m³/s.`:'HydroRIVERS river network.';
 $('peak-count').textContent=data.peaks?.count?`${(data.peaks.selected||0).toLocaleString()} selected / ${data.peaks.count.toLocaleString()} mapped summits.`:'Mapped mountain summits.';
 const credits=[];
 for(const name of ['satellite','streets','semantic'])if(s.config.layers[name].enabled&&data.sources?.[name])credits.push(name==='satellite'&&s.config.layers.satellite.month!==0&&data.seasonal?.credit?data.seasonal:data.sources[name]);
 if(s.config.layers.rivers.enabled&&data.rivers?.source)credits.push({credit:data.rivers.source,url:data.rivers.url});
 if(s.config.layers.divides.enabled&&data.divides?.source)credits.push({credit:data.divides.source,url:data.divides.url});
 if(s.config.layers.peaks.enabled&&data.peaks?.source)credits.push({credit:data.peaks.source,url:data.peaks.url});
 const signature=JSON.stringify(credits);
 if($('map-attribution').dataset.credits!==signature){
  $('map-attribution').replaceChildren();
  for(const [i,credit] of credits.entries()){
   if(i)$('map-attribution').append(' · ');
   const a=document.createElement('a');a.textContent=credit.credit;a.href=credit.url;a.target='_blank';a.rel='noopener';$('map-attribution').append(a);
  }
  $('map-attribution').dataset.credits=signature;
 }
}
for(const name of ['white','satellite','streets','semantic','contours','rivers','peaks','divides']){
 const enabled=$(`layer-${name}-enabled`);
 enabled.addEventListener('click',event=>event.stopPropagation());
 enabled.addEventListener('change',()=>api('/api/config',{pattern:'sun',layers:{[name]:{enabled:enabled.checked}}}).catch(e=>showError(e.message)));
 for(const input of document.querySelectorAll(`[id^="layer-${name}-"]`)){
  const key=input.id.slice(`layer-${name}-`.length);
  if(key==='enabled'||input.tagName==='OUTPUT')continue;
  const send=persist=>{
   let v=input.type==='checkbox'?input.checked:input.type==='color'?input.value:input.tagName==='SELECT'?(key==='single'?input.value==='true':key==='month'?Number(input.value):input.value):Number(input.value);
   const patch={[key]:v};
   if(name==='peaks'){
    if(key==='min_elevation'&&v>Number($('layer-peaks-max_elevation').value)){$('layer-peaks-max_elevation').value=v;patch.max_elevation=v;}
    if(key==='max_elevation'&&v<Number($('layer-peaks-min_elevation').value)){$('layer-peaks-min_elevation').value=v;patch.min_elevation=v;}
   }
   layerReadout(name,key);liveUpdate({pattern:'sun',layers:{[name]:patch}},persist);
  };
  input.addEventListener('input',()=>send(false));input.addEventListener('change',()=>send(true));
 }
}
$('layer-track-enabled').addEventListener('click',event=>event.stopPropagation());
$('layer-track-enabled').addEventListener('change',()=>api('/api/config',{pattern:'sun',track:{enabled:$('layer-track-enabled').checked}}).catch(e=>showError(e.message)));
$('layer-shadow-enabled').addEventListener('change',()=>api('/api/config',{pattern:'sun',sun:{enabled:$('layer-shadow-enabled').checked}}).catch(e=>showError(e.message)));
action('show-layers',()=>api('/api/config',{pattern:'sun'}));
for(const [event,persist] of [['input',false],['change',true]])$('track-width').addEventListener(event,()=>{
 $('track-width-value').textContent=`${value('track-width')} px`;liveUpdate({pattern:'sun',track:{width:value('track-width')}},persist);
});

action('rivers-replay',()=>api('/api/rivers/replay',{}));

for(const [id,key] of [['shadow-floor-enabled','shadow_floor'],['shadow-normalize','normalize']])$(id).addEventListener('change',()=>api('/api/config',{sun:{[key]:$(id).checked}}).catch(e=>showError(e.message)));
for(const [event,persist] of [['input',false],['change',true]])$('shadow-floor-amount').addEventListener(event,()=>{const n=value('shadow-floor-amount');$('shadow-floor-value').textContent=`${Math.round(n*100)}%`;liveUpdate({sun:{shadow_floor_amount:n}},persist);});
