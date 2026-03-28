/**
 * Regenerates web/public/data/teams.json from data/MTeams.csv and data/WTeams.csv.
 * Run from repo root: node scripts/generate_teams_json.mjs
 * Or from web/: npm run generate:teams
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(__dirname, "..");

function readTeams(csvPath) {
  const text = fs.readFileSync(csvPath, "utf8");
  const lines = text.trim().split(/\r?\n/);
  const header = lines[0].split(",");
  const idIdx = header.indexOf("TeamID");
  const nameIdx = header.indexOf("TeamName");
  if (idIdx < 0 || nameIdx < 0) {
    throw new Error(`Missing TeamID/TeamName in ${csvPath}`);
  }
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(",");
    const id = Number.parseInt(parts[idIdx], 10);
    const name = String(parts[nameIdx] ?? "").trim();
    if (Number.isFinite(id)) {
      rows.push({ id, name });
    }
  }
  rows.sort((a, b) => a.name.localeCompare(b.name) || a.id - b.id);
  return rows;
}

const mPath = path.join(root, "data", "MTeams.csv");
const wPath = path.join(root, "data", "WTeams.csv");
if (!fs.existsSync(mPath) || !fs.existsSync(wPath)) {
  console.error(`Need ${mPath} and ${wPath}`);
  process.exit(1);
}

const payload = {
  generatedAt: new Date().toISOString(),
  sources: { M: "MTeams.csv", W: "WTeams.csv" },
  M: readTeams(mPath),
  W: readTeams(wPath),
};

const out = path.join(root, "web", "public", "data", "teams.json");
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, JSON.stringify(payload, null, 2) + "\n", "utf8");
console.log(
  `Wrote ${out} (${payload.M.length} men, ${payload.W.length} women teams)`,
);
