// Fetches a week of Admiralty tide predictions for the port nearest Patch
// and writes tides.json for index.html to read. Needs Node 20 or newer.
//   UKHO_KEY=yourkey node fetch-tides.mjs
import { writeFile } from 'node:fs/promises';

const KEY = process.env.UKHO_KEY;
if (!KEY) { console.error('Set UKHO_KEY to your Admiralty subscription key.'); process.exit(1); }

const LAT = 52.104699, LON = -4.681918;
const BASE = 'https://admiraltyapi.azure-api.net/uktidalapi/api/V1';

async function get(path) {
  const r = await fetch(BASE + path, { headers: { 'Ocp-Apim-Subscription-Key': KEY } });
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status} ${await r.text()}`);
  return r.json();
}

// Set STATION_ID to skip the lookup and force a particular port.
let station = process.env.STATION_ID ? { id: process.env.STATION_ID, name: process.env.STATION_ID } : null;
if (!station) {
  const list = await get('/Stations');
  for (const f of list.features ?? list) {
    const [lon, lat] = f.geometry.coordinates;
    const d = Math.hypot(lat - LAT, (lon - LON) * Math.cos(LAT * Math.PI / 180));
    if (!station || d < station.d) station = { d, id: f.properties.Id, name: f.properties.Name };
  }
}

const events = await get(`/Stations/${station.id}/TidalEvents?duration=7`);
await writeFile('tides.json', JSON.stringify(
  { station: station.name, id: station.id, fetched: new Date().toISOString(), events }, null, 1));
console.log(`${station.name} (${station.id}): ${events.length} tidal events saved`);
