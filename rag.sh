#!/usr/bin/env bash
# =============================================================================
# rag.sh — IW RAG stack management
#
# iw-rag is fully containerized (no host processes of its own). This wraps the
# Docker Compose layering the Makefile encodes, plus lifecycle for the guard
# sidecars and the GraphRAG store — start/stop/restart/status, individually or
# in batch.
#
# Usage (interactive):  ./rag.sh
# Usage (CLI):          ./rag.sh up
#                       ./rag.sh guards up
#                       ./rag.sh status
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# LOAD .env  (so POSTGRES_*, HF_TOKEN, IW_RAG__* resolve for compose + psql)
# =============================================================================

load_env() {
  [[ -f "$SCRIPT_DIR/.env" ]] || return 0
  local tmp_env
  tmp_env=$(mktemp)
  tr -d '\r' < "$SCRIPT_DIR/.env" > "$tmp_env"
  set -a
  # shellcheck disable=SC1090
  source "$tmp_env"
  set +a
  rm -f "$tmp_env"
}
load_env

# =============================================================================
# CONFIG
# =============================================================================

EDGE_PORT="8088"          # Caddy edge → the one entrypoint
PHOENIX_PORT="6006"
DB_PORT="5432"
GRAPH_PORT="5434"
DB_USER="${POSTGRES_USER:-iw_rag}"
DB_NAME="${POSTGRES_DB:-iw_rag}"

# Compose layering (mirrors the Makefile). --project-directory . so the repo root
# is the project dir: root .env is read and ./docker/... bind paths resolve.
COMPOSE_BASE=(docker compose --project-directory . -f docker/compose.yaml -f docker/compose.gpu.yaml)
GUARDS_FILES=(-f docker/compose.guards-sidecars.yaml -f docker/compose.guards.yaml)
GRAPH_FILES=(-f docker/compose.graph.yaml)
BENCH_FILES=(-f docker/compose.benchmark.yaml)

CORE_SERVICES=(postgres vllm infinity phoenix api caddy)
CORE_CONTAINERS=(iw-rag-postgres-1 iw-rag-vllm-1 iw-rag-infinity-1 iw-rag-phoenix-1 iw-rag-api-1 iw-rag-caddy-1)
GUARD_CONTAINERS=(injguard iw-rag-vllm-guard)
GRAPH_CONTAINER=iw-rag-graph-pg

# =============================================================================
# COLORS & OUTPUT HELPERS
# =============================================================================

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'
BLUE=$'\033[0;34m'; CYAN=$'\033[0;36m'; BOLD=$'\033[1m'; DIM=$'\033[2m'; NC=$'\033[0m'

print_ok()   { echo -e "  ${GREEN}✓${NC}  $*"; }
print_err()  { echo -e "  ${RED}✗${NC}  $*"; }
print_warn() { echo -e "  ${YELLOW}!${NC}  $*"; }
print_info() { echo -e "  ${CYAN}→${NC}  $*"; }
print_dim()  { echo -e "  ${DIM}$*${NC}"; }
print_header() { echo ""; echo -e "${BOLD}${BLUE}=== $1 ===${NC}"; echo ""; }

confirm() {
  local prompt="${1:-Are you sure?}" answer
  echo ""
  read -r -p "  ${YELLOW}${prompt} [y/N]${NC} " answer
  case "$answer" in [yY][eE][sS]|[yY]) return 0 ;; *) return 1 ;; esac
}

# =============================================================================
# DOCKER / PROBE UTILITIES
# =============================================================================

# cstate <name> — running|exited|absent
cstate() {
  docker inspect -f '{{.State.Status}}' "$1" 2>/dev/null || echo absent
}
# chealth <name> — healthy|unhealthy|starting|<empty if no healthcheck>
chealth() {
  docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$1" 2>/dev/null
}
container_exists() { docker inspect "$1" &>/dev/null; }
# compose_managed <name> — true if the container carries the iw-rag compose label
compose_managed() {
  [[ "$(docker inspect -f '{{index .Config.Labels "com.docker.compose.project"}}' "$1" 2>/dev/null)" == "iw-rag" ]]
}
http_ok() {
  local code
  code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 "$1" 2>/dev/null) || return 1
  [[ "$code" =~ ^[23] ]]
}

# A legacy (pre-rag.sh) hand-run guard/graph container blocks compose from
# creating a same-named, label-managed one. Retire it once — its state lives in a
# shared model cache (guards) or an external volume (graph), so nothing is lost.
retire_legacy() {
  local name="$1"
  if container_exists "$name" && ! compose_managed "$name"; then
    print_warn "Retiring legacy hand-run container '$name' so compose can manage it (data preserved)"
    docker rm -f "$name" >/dev/null 2>&1 || true
  fi
}

print_service_line() {
  # print_service_line <pretty> <container>
  local pretty="$1" name="$2" state health
  state=$(cstate "$name"); health=$(chealth "$name")
  if [[ "$state" == running ]]; then
    if [[ -n "$health" && "$health" != healthy ]]; then
      print_warn "$pretty: running (${health})"
    else
      print_ok "$pretty: running"
    fi
  elif [[ "$state" == absent ]]; then
    print_err "$pretty: not created"
  else
    print_err "$pretty: $state"
  fi
}

# =============================================================================
# CORE STACK  (postgres, vllm, infinity, phoenix, api, caddy)
# =============================================================================

cmd_core() {
  local sub="${1:-help}"; shift || true
  case "$sub" in
    up|start)
      print_info "Starting core stack (postgres · vllm · infinity · phoenix · api · caddy)..."
      "${COMPOSE_BASE[@]}" up -d "$@" || return 1
      print_ok "Core stack up — edge: http://localhost:${EDGE_PORT}  ·  phoenix: http://localhost:${PHOENIX_PORT}"
      ;;
    down)
      # `compose down` removes ALL iw-rag-project containers (guards + graph too).
      print_info "Tearing down the whole iw-rag project (core + any guards/graph)..."
      "${COMPOSE_BASE[@]}" "${GUARDS_FILES[@]}" "${GRAPH_FILES[@]}" down "$@"
      print_ok "Project down"
      ;;
    stop)
      print_info "Stopping core services (containers kept)..."
      "${COMPOSE_BASE[@]}" stop "${CORE_SERVICES[@]}"
      print_ok "Core stopped"
      ;;
    restart)
      "${COMPOSE_BASE[@]}" restart "${@:-${CORE_SERVICES[@]}}"
      print_ok "Core restarted"
      ;;
    rebuild-api)
      print_info "Rebuilding + recreating the api container..."
      docker build -t iw-rag-api -f docker/api.Dockerfile . || return 1
      "${COMPOSE_BASE[@]}" up -d --no-build --no-deps --force-recreate api
      print_ok "api redeployed"
      ;;
    status)
      local i
      for i in "${!CORE_SERVICES[@]}"; do
        print_service_line "${CORE_SERVICES[$i]}" "${CORE_CONTAINERS[$i]}"
      done
      ;;
    logs)
      "${COMPOSE_BASE[@]}" logs -f --tail "${LOGS_TAIL:-50}" "$@"
      ;;
    *)
      echo "  Usage: $0 core {up|down|stop|restart|rebuild-api|status|logs [svc...]}"
      ;;
  esac
}

# =============================================================================
# GUARDRAILS  (injguard:8003 + iw-rag-vllm-guard:8004 + api guard wiring)
# =============================================================================

cmd_guards() {
  local sub="${1:-help}"; shift || true
  case "$sub" in
    up|on)
      retire_legacy injguard
      retire_legacy iw-rag-vllm-guard
      print_info "Starting guard sidecars (injection classifier + Granite Guardian)..."
      "${COMPOSE_BASE[@]}" "${GUARDS_FILES[@]}" up -d injguard iw-rag-vllm-guard || return 1
      print_info "Recreating api WITH the guards overlay (endpoints wired, fail-closed)..."
      "${COMPOSE_BASE[@]}" "${GUARDS_FILES[@]}" up -d --no-build --no-deps --force-recreate api || return 1
      print_ok "Guards ON. vllm-guard cold-starts the 8B judge — watch: $0 guards logs"
      print_dim "Granite Guardian can take several minutes to become healthy on first load."
      ;;
    down|off)
      print_info "Stopping + removing guard sidecars..."
      "${COMPOSE_BASE[@]}" "${GUARDS_FILES[@]}" rm -sf injguard iw-rag-vllm-guard
      print_info "Recreating api WITHOUT guards (GUARDRAILS disabled to avoid fail-closed)..."
      IW_RAG__GUARDRAILS__ENABLED=false "${COMPOSE_BASE[@]}" up -d --no-build --no-deps --force-recreate api || return 1
      print_ok "Guards OFF (api answers the core grounded path; guard endpoints not called)"
      ;;
    restart)
      cmd_guards down; cmd_guards up
      ;;
    status)
      print_service_line "injguard (injection :8003)" injguard
      print_service_line "iw-rag-vllm-guard (Granite :8004)" iw-rag-vllm-guard
      if http_ok "http://localhost:8004/health"; then
        print_ok "Granite Guardian: serving on :8004"
      else
        print_warn "Granite Guardian: :8004 not answering (cold-starting or off)"
      fi
      ;;
    logs)
      "${COMPOSE_BASE[@]}" "${GUARDS_FILES[@]}" logs -f --tail "${LOGS_TAIL:-50}" injguard iw-rag-vllm-guard
      ;;
    *)
      echo "  Usage: $0 guards {up|down|restart|status|logs}"
      ;;
  esac
}

# =============================================================================
# GRAPHRAG STORE  (iw-rag-graph-pg :5434 — AGE-on-PG18)
# =============================================================================

cmd_graph() {
  local sub="${1:-help}"; shift || true
  case "$sub" in
    up|start)
      retire_legacy "$GRAPH_CONTAINER"
      print_info "Starting GraphRAG store (AGE-on-PG18) on :${GRAPH_PORT}..."
      "${COMPOSE_BASE[@]}" "${GRAPH_FILES[@]}" up -d graph-pg || return 1
      print_ok "graph-pg up — postgresql://${DB_USER}@localhost:${GRAPH_PORT}/${DB_NAME}"
      ;;
    down|stop)
      print_info "Stopping GraphRAG store..."
      "${COMPOSE_BASE[@]}" "${GRAPH_FILES[@]}" stop graph-pg 2>/dev/null \
        || docker stop "$GRAPH_CONTAINER" >/dev/null 2>&1
      print_ok "graph-pg stopped"
      ;;
    restart)
      cmd_graph down; cmd_graph up
      ;;
    status)
      print_service_line "graph-pg (AGE :${GRAPH_PORT})" "$GRAPH_CONTAINER"
      if ! compose_managed "$GRAPH_CONTAINER" && container_exists "$GRAPH_CONTAINER"; then
        print_warn "graph-pg is a legacy hand-run container — run '$0 graph up' to adopt it into compose"
      fi
      ;;
    psql)
      print_info "psql → graph store (${DB_USER}@localhost:${GRAPH_PORT}/${DB_NAME})..."
      docker exec -it "$GRAPH_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME"
      ;;
    logs)
      docker logs -f --tail "${LOGS_TAIL:-50}" "$GRAPH_CONTAINER"
      ;;
    *)
      echo "  Usage: $0 graph {up|down|restart|status|psql|logs}"
      ;;
  esac
}

# =============================================================================
# DATABASE / OPS  (migrations, psql, spike, health)
# =============================================================================

cmd_db() {
  local sub="${1:-help}"; shift || true
  case "$sub" in
    migrate)
      print_info "Applying Alembic migrations (alembic upgrade head)..."
      uv run alembic upgrade head && print_ok "Migrations up to date"
      ;;
    revision)
      local msg="${1:-}"
      [[ -z "$msg" ]] && { print_err "Usage: $0 db revision <message>"; return 1; }
      print_info "Authoring a PENDING-rebased migration (make migration-pending)..."
      make migration-pending MSG="$msg"
      ;;
    psql)
      print_info "psql → main store (${DB_USER}@localhost:${DB_PORT}/${DB_NAME})..."
      docker exec -it iw-rag-postgres-1 psql -U "$DB_USER" -d "$DB_NAME"
      ;;
    *)
      echo "  Usage: $0 db {migrate|revision <msg>|psql}"
      ;;
  esac
}

cmd_spike() {
  print_info "Running the VRAM co-residence gating spike (make spike)..."
  make spike
}

cmd_health() {
  print_info "Deep health probe (DB + extensions · vLLM · infinity embed/rerank)..."
  uv run python scripts/health_probe.py
}

# =============================================================================
# OPTIONAL OVERLAYS (thin pass-throughs)
# =============================================================================

cmd_bench() {
  local sub="${1:-help}"; shift || true
  case "$sub" in
    up)   print_info "Starting Benchmark Studio (llama-swap + swap-controller)...";
          "${COMPOSE_BASE[@]}" "${BENCH_FILES[@]}" up -d ;;
    down) "${COMPOSE_BASE[@]}" "${BENCH_FILES[@]}" stop llama-swap swap-controller ;;
    *)    echo "  Usage: $0 bench {up|down}" ;;
  esac
}

cmd_visual() {
  local sub="${1:-help}"; shift || true
  if [[ ! -f docker/compose.vlm.yaml ]]; then
    print_err "docker/compose.vlm.yaml is missing — visual/VLM mode cannot start on this checkout."
    print_dim "Restore the overlay (Qwen2.5-VL caption server) before using 'visual up'."
    return 1
  fi
  case "$sub" in
    up)   make visual-up ;;
    down) make visual-down ;;
    *)    echo "  Usage: $0 visual {up|down}" ;;
  esac
}

cmd_e2e() {
  print_info "The E2E stack (docker/compose.e2e.yaml) is daemon-orchestrated per worktree."
  print_dim "It needs COMPOSE_PROJECT_NAME + E2E_*_PORT env from orch.daemon.browser_env."
  print_dim "Manual bring-up:  scripts/e2e_up.sh   ·   tear-down:  scripts/e2e_down.sh"
}

# =============================================================================
# COMPOUND  (start / stop / restart / status)
# =============================================================================

cmd_start() {
  local with_guards=false with_graph=false
  for a in "$@"; do
    case "$a" in
      --guards) with_guards=true ;;
      --graph)  with_graph=true ;;
      --all)    with_guards=true; with_graph=true ;;
    esac
  done
  print_header "Starting IW RAG"
  cmd_core up || return 1
  $with_graph  && { echo ""; cmd_graph up; }
  $with_guards && { echo ""; cmd_guards up; }
  echo ""
  print_ok "Started.  Edge → http://localhost:${EDGE_PORT}   (analyst/analyst · hr/hr)"
}

cmd_stop() {
  print_header "Stopping IW RAG"
  cmd_core down
  echo ""
  print_ok "All iw-rag containers stopped"
}

cmd_status() {
  print_header "IW RAG — Status"
  echo -e "  ${BOLD}Core${NC}"
  cmd_core status
  echo ""
  echo -e "  ${BOLD}Guardrails${NC}"
  cmd_guards status
  echo ""
  echo -e "  ${BOLD}GraphRAG${NC}"
  cmd_graph status
  echo ""
  echo -e "  ${BOLD}Endpoints${NC}"
  if http_ok "http://localhost:${EDGE_PORT}/"; then
    print_ok "Edge (Caddy): http://localhost:${EDGE_PORT}"
  else
    print_err "Edge (Caddy): http://localhost:${EDGE_PORT} not answering"
  fi
  http_ok "http://localhost:${PHOENIX_PORT}/" \
    && print_ok "Phoenix: http://localhost:${PHOENIX_PORT}" \
    || print_warn "Phoenix: http://localhost:${PHOENIX_PORT} not answering"
  echo ""
}

# single-line status for the menu header
get_status_line() {
  local core_up=0 c
  for c in "${CORE_CONTAINERS[@]}"; do [[ "$(cstate "$c")" == running ]] && core_up=$((core_up+1)); done
  local core_badge; [[ $core_up -eq ${#CORE_CONTAINERS[@]} ]] && core_badge="${GREEN}▲ core ${core_up}/${#CORE_CONTAINERS[@]}${NC}" || core_badge="${YELLOW}◆ core ${core_up}/${#CORE_CONTAINERS[@]}${NC}"
  [[ $core_up -eq 0 ]] && core_badge="${RED}▼ core 0/${#CORE_CONTAINERS[@]}${NC}"
  local g="${RED}▼ guards${NC}"; [[ "$(cstate iw-rag-vllm-guard)" == running ]] && g="${GREEN}▲ guards${NC}"
  local gr="${RED}▼ graph${NC}"; [[ "$(cstate "$GRAPH_CONTAINER")" == running ]] && gr="${GREEN}▲ graph${NC}"
  echo -e "  ${core_badge}    ${g}    ${gr}"
}

cmd_logs() { cmd_core logs "$@"; }

# =============================================================================
# INTERACTIVE MENUS
# =============================================================================

_pause() { echo ""; read -r -p "  Press Enter to continue..."; }

menu_guards() {
  while true; do
    print_header "Guardrails"; cmd_guards status; echo ""
    echo -e "  ${CYAN}1)${NC} Turn ON   (sidecars + api fail-closed)"
    echo -e "  ${CYAN}2)${NC} Turn OFF  (stop sidecars + api guards-disabled)"
    echo -e "  ${CYAN}3)${NC} Restart"
    echo -e "  ${CYAN}4)${NC} Logs (live)"
    echo -e "  ${CYAN}0)${NC} Back"; echo ""
    read -r -p "  Choice: " c
    case "$c" in
      1) cmd_guards up ;; 2) cmd_guards down ;; 3) cmd_guards restart ;;
      4) cmd_guards logs ;; 0|"") return ;; *) print_warn "Invalid choice" ;;
    esac
    _pause
  done
}

menu_graph() {
  while true; do
    print_header "GraphRAG store"; cmd_graph status; echo ""
    echo -e "  ${CYAN}1)${NC} Start"
    echo -e "  ${CYAN}2)${NC} Stop"
    echo -e "  ${CYAN}3)${NC} Restart"
    echo -e "  ${CYAN}4)${NC} psql shell"
    echo -e "  ${CYAN}5)${NC} Logs (live)"
    echo -e "  ${CYAN}0)${NC} Back"; echo ""
    read -r -p "  Choice: " c
    case "$c" in
      1) cmd_graph up ;; 2) cmd_graph down ;; 3) cmd_graph restart ;;
      4) cmd_graph psql ;; 5) cmd_graph logs ;; 0|"") return ;; *) print_warn "Invalid choice" ;;
    esac
    _pause
  done
}

menu_core() {
  while true; do
    print_header "Core stack"; cmd_core status; echo ""
    echo -e "  ${CYAN}1)${NC} Up"
    echo -e "  ${CYAN}2)${NC} Stop"
    echo -e "  ${CYAN}3)${NC} Restart"
    echo -e "  ${CYAN}4)${NC} Rebuild + redeploy api"
    echo -e "  ${CYAN}5)${NC} Logs (live, all core)"
    echo -e "  ${CYAN}0)${NC} Back"; echo ""
    read -r -p "  Choice: " c
    case "$c" in
      1) cmd_core up ;; 2) cmd_core stop ;; 3) cmd_core restart ;;
      4) cmd_core rebuild-api ;; 5) cmd_core logs ;; 0|"") return ;; *) print_warn "Invalid choice" ;;
    esac
    _pause
  done
}

menu_ops() {
  while true; do
    print_header "Database / Ops"
    echo -e "  ${CYAN}1)${NC} Apply migrations  (alembic upgrade head)"
    echo -e "  ${CYAN}2)${NC} Author migration  (make migration-pending)"
    echo -e "  ${CYAN}3)${NC} psql → main store"
    echo -e "  ${CYAN}4)${NC} VRAM spike  (gating)"
    echo -e "  ${CYAN}5)${NC} Deep health probe"
    echo -e "  ${CYAN}0)${NC} Back"; echo ""
    read -r -p "  Choice: " c
    case "$c" in
      1) cmd_db migrate ;;
      2) read -r -p "  Migration message: " m; cmd_db revision "$m" ;;
      3) cmd_db psql ;; 4) cmd_spike ;; 5) cmd_health ;;
      0|"") return ;; *) print_warn "Invalid choice" ;;
    esac
    _pause
  done
}

menu_main() {
  while true; do
    clear
    echo -e "${BOLD}${BLUE}"
    echo "  ╔══════════════════════════════════════╗"
    echo "  ║   IW RAG                             ║"
    echo "  ║   Local Agentic RAG Stack            ║"
    echo "  ╚══════════════════════════════════════╝"
    echo -e "${NC}"
    get_status_line; echo ""
    echo -e "  ${CYAN}1)${NC} Start core         (postgres → models → api → caddy)"
    echo -e "  ${CYAN}2)${NC} Start everything   (core + graph + guards)"
    echo -e "  ${CYAN}3)${NC} Stop all"
    echo -e "  ${CYAN}4)${NC} Restart core"
    echo -e "  ${CYAN}5)${NC} Status"
    echo ""
    echo -e "  ${CYAN}6)${NC} Core stack"
    echo -e "  ${CYAN}7)${NC} Guardrails"
    echo -e "  ${CYAN}8)${NC} GraphRAG store"
    echo -e "  ${CYAN}9)${NC} Database / Ops"
    echo -e "  ${CYAN}l)${NC} Logs (all core, live)"
    echo ""
    echo -e "  ${CYAN}0)${NC} Exit"; echo ""
    read -r -p "  Choice: " choice
    case "$choice" in
      1) cmd_start ;;
      2) cmd_start --all ;;
      3) cmd_stop ;;
      4) cmd_core restart ;;
      5) cmd_status ;;
      6) menu_core ;;
      7) menu_guards ;;
      8) menu_graph ;;
      9) menu_ops ;;
      l|L) cmd_logs ;;
      0|q|Q|exit|quit) echo ""; exit 0 ;;
      *) print_warn "Invalid choice" ;;
    esac
    _pause
  done
}

# =============================================================================
# CLI ENTRY POINT
# =============================================================================

usage() {
  cat <<EOF

${BOLD}rag.sh${NC} — IW RAG stack management

${BOLD}USAGE${NC}
  ./rag.sh                          Interactive menu
  ./rag.sh <command> [args]         CLI mode

${BOLD}COMPOUND${NC}
  up | start [--guards|--graph|--all]   Start core (and optionally guards/graph)
  stop                                   Tear down all iw-rag containers
  restart                                Restart core services
  status                                 Full status across core / guards / graph

${BOLD}CORE${NC}        postgres · vllm · infinity · phoenix · api · caddy
  core up|down|stop|restart|status|logs [svc...]
  core rebuild-api                          Rebuild the api image + recreate it

${BOLD}GUARDRAILS${NC}  injguard:8003 · iw-rag-vllm-guard:8004
  guards up|down|restart|status|logs     up wires the api fail-closed; down disables

${BOLD}GRAPHRAG${NC}    iw-rag-graph-pg:5434 (AGE-on-PG18)
  graph up|down|restart|status|psql|logs

${BOLD}DATABASE / OPS${NC}
  db migrate | db revision <msg> | db psql
  spike                                  VRAM co-residence gating spike
  health                                 Deep dependency health probe

${BOLD}OPTIONAL${NC}
  bench up|down                          Benchmark Studio (llama-swap)
  visual up|down                         VLM caption server (needs compose.vlm.yaml)
  e2e                                    Notes on the daemon-driven E2E stack

${BOLD}EXAMPLES${NC}
  ./rag.sh up
  ./rag.sh start --all
  ./rag.sh guards up
  ./rag.sh status
  ./rag.sh core logs api

EOF
}

CMD="${1:-}"; shift || true
case "$CMD" in
  "")                 menu_main ;;
  up|start)           cmd_start "$@" ;;
  stop|down)          cmd_stop ;;
  restart)            cmd_core restart "$@" ;;
  status)             cmd_status ;;
  logs)               cmd_logs "$@" ;;
  core)               cmd_core "$@" ;;
  guards)             cmd_guards "$@" ;;
  graph)              cmd_graph "$@" ;;
  db)                 cmd_db "$@" ;;
  spike)              cmd_spike ;;
  health)             cmd_health ;;
  bench)              cmd_bench "$@" ;;
  visual)             cmd_visual "$@" ;;
  e2e)                cmd_e2e ;;
  help|--help|-h)     usage ;;
  *) print_err "Unknown command: $CMD"; usage; exit 1 ;;
esac
