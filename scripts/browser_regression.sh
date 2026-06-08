#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# LexIntel browser regression suite (agent-browser as the QA tester).
#
# Drives the running frontend through every route + the matter workspace tabs +
# the Ask-AI flow, asserting each page renders and produces NO console errors.
# Re-runnable: `scripts/browser_regression.sh [FRONTEND_URL] [MATTER_ID]`.
#
# Requires: agent-browser on PATH; frontend + backend running.
# Exit code 0 = all checks passed, 1 = one or more failures.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

FE="${1:-http://localhost:3100}"
MID="${2:-495537db-621e-44c9-ad86-bc3fb0540ec8}"   # a ready matter
export AGENT_BROWSER_SCREENSHOT_DIR="${AGENT_BROWSER_SCREENSHOT_DIR:-/tmp/lex_regression}"
mkdir -p "$AGENT_BROWSER_SCREENSHOT_DIR"

pass=0; fail=0
ab(){ agent-browser "$@" 2>/dev/null; }
# agent-browser `eval` returns JSON-encoded values (e.g. "true"); strip quotes/space.
abeval(){ agent-browser eval "$1" 2>/dev/null | tail -1 | tr -d '"' | tr -d '[:space:]'; }
# console errors excluding known-benign library noise
console_errors(){ ab console 2>&1 | grep -iE "error|exception|cannot read|undefined is not|is not a function|chunkloaderror|failed to fetch" \
  | grep -viE "scroll offset|non-static position|favicon|404 \(Not Found\).*favicon" | head -5; }
check(){ # check "<label>" "<signal-present-bool>"
  local label="$1" ok="$2" errs; errs="$(console_errors)"
  if [ "$ok" = "true" ] && [ -z "$errs" ]; then echo "  PASS  $label"; pass=$((pass+1));
  else echo "  FAIL  $label"; [ -n "$errs" ] && echo "        console: ${errs//$'\n'/ | }"; fail=$((fail+1)); fi
}

echo "== LexIntel browser regression ($FE) =="
agent-browser close >/dev/null 2>&1; sleep 1

# Auth (demo token) then land
ab open "$FE/login" >/dev/null; ab wait 2000 >/dev/null
ab eval "localStorage.setItem('lexintel_token','demo-token');'ok'" >/dev/null

echo "-- top-level routes --"
for route in "" matters dashboard precedents team billing settings; do
  ab console --clear >/dev/null
  ab open "$FE/$route" >/dev/null; ab wait 2500 >/dev/null
  nonblank="$(abeval "(document.body.innerText.trim().length>20).toString()")"
  check "/$route renders" "$([ "$nonblank" = "true" ] && echo true || echo false)"
done

echo "-- matter workspace tabs --"
ab open "$FE/matters/$MID" >/dev/null; ab wait 4000 >/dev/null
ready="$(abeval "(document.body.innerText.includes('Ready')||document.body.innerText.includes('Ask AI')).toString()")"
check "matter loads" "$([ "$ready" = "true" ] && echo true || echo false)"
for tab in "Documents" "Contract Review" "Draft Assistant" "Citation Graph" "Audit Log" "Ask AI"; do
  ab console --clear >/dev/null
  ab find text "$tab" click >/dev/null; ab wait 2500 >/dev/null
  check "tab: $tab" true
done

echo "-- Ask AI flow --"
ab console --clear >/dev/null
ref="$(ab snapshot 2>&1 | grep -iE 'textbox "Ask a question' | grep -oE 'e[0-9]+' | head -1)"
if [ -n "$ref" ]; then
  ab type "@$ref" "What are the payment terms?" >/dev/null
  ab press Enter >/dev/null
  ab wait 22000 >/dev/null
  body="$(ab eval "document.body.innerText" | tail -1)"
  # success = a rendered assistant message (answer text OR a graceful error message), never raw 'None'/stacktrace
  graceful="$(echo "$body" | grep -ciE "No (relevant|documents|chunks)|temporarily unavailable|couldn't|payment|fee|delaware|liable")"
  rawnone="$(echo "$body" | grep -cE ': None|Traceback|undefined')"
  check "Ask AI returns a rendered message (no raw None/trace)" "$([ "$graceful" -ge 1 ] && [ "$rawnone" -eq 0 ] && echo true || echo false)"
else
  echo "  FAIL  Ask AI input not found"; fail=$((fail+1))
fi
ab screenshot "$AGENT_BROWSER_SCREENSHOT_DIR/askai.png" >/dev/null 2>&1
agent-browser close >/dev/null 2>&1

echo "== RESULT: $pass passed, $fail failed =="
[ "$fail" -eq 0 ]
