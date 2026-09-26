#!/bin/bash
# GTS Public CA rate-limit test. Startup script for a throwaway e2-micro.
#
# Question it answers: does Google Trust Services (Cloud Public CA, ACME with
# External Account Binding) tolerate the enclave's certificate pattern, which
# is a fresh EAB, a fresh ACME account, a fresh key and a fresh certificate
# for the same hostname on every boot? Let's Encrypt caps that at 5 per
# hostname per week. Persisting a certificate is not an option: any storage
# the operator owns, the operator can read.
#
# Phases, one JSON line per iteration on the serial console (prefix GTSTEST):
#   1. staging     N_STAGING lego issuances, each from a new account, 20 s apart
#   2. shortlived  one staging issuance with notAfter = now + 7 d
#   3. caddy       N_CADDY runs of the enclave's Caddy issuer stanza against
#                  staging, storage wiped between runs, EAB via the eab
#                  subdirective
#   4. production  N_PROD lego issuances on the production directory; the last
#                  certificate is left serving a static page on :443
#
# Launch (from masseuse-video-tee/README.md, "Certificate authority"):
#   gcloud compute instances create gts-acme-test --project prod-masseuse-video-tee \
#     --zone us-central1-a --machine-type e2-micro --image-family debian-12 \
#     --image-project debian-cloud --subnet masseuse-video-tee-us-central1 \
#     --tags masseuse-video-tee --address gts-acme-test \
#     --service-account gts-acme-test@prod-masseuse-video-tee.iam.gserviceaccount.com \
#     --scopes cloud-platform --metadata-from-file startup-script=tools/gts-acme-test.sh \
#     --metadata gts-host=acme-test.masseuse.ai
#   gcloud compute scp lego-ku gts-acme-test:/tmp/ --tunnel-through-iap   # patched lego, see below
#   gcloud compute ssh gts-acme-test --tunnel-through-iap -- sudo install -m 755 /tmp/lego-ku /usr/local/bin/lego-ku
# Watch: gcloud compute instances get-serial-port-output gts-acme-test --zone us-central1-a | grep GTSTEST
# Re-run after a metadata change: `sudo systemctl restart google-startup-scripts
# --no-block` (the unit is oneshot; without --no-block the restart blocks for
# the whole run), then check `pgrep -af startup-script` shows one copy.
#
# Runs as root under google-startup-scripts. Nothing here is reused by the
# enclave except the EAB request shape, which tee/entrypoint.sh will repeat
# with a WIF token instead of the metadata server's.
#
# lego needs a patch to pass GTS validation (found 2026-09-08, see the README):
# its TLS-ALPN-01 challenge certificate carries KeyUsage = keyEncipherment
# only, and the BoringSSL-based validators GTS uses for its remote MPIC
# perspectives refuse to let a certificate without digitalSignature sign a
# TLS 1.3 CertificateVerify (alert illegal_parameter). Let's Encrypt's
# Go-based validators do not check key usage, which is why nobody noticed.
# Caddy's challenge certificate is fine. The patch is one line in
# certcrypto/crypto.go (generateDerCert):
#   KeyUsage: x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
# Build: copy the v5.4.1 module, apply, `GOOS=linux GOARCH=amd64 CGO_ENABLED=0
# go build -o lego-ku .`, and either scp it to /usr/local/bin/lego-ku on the
# VM before the A record lands or serve it at the URL in the gts-lego-url
# metadata attribute. Without it every lego iteration fails with
# "connection :: No response from <host>" and the run says nothing about
# rate limits.
set -u
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

MD=http://metadata.google.internal/computeMetadata/v1
md() { curl -fsS -H 'Metadata-Flavor: Google' "$MD/$1" 2>/dev/null; }
attr() { md "instance/attributes/$1" || echo "$2"; }

HOST=$(attr gts-host acme-test.masseuse.ai)
EMAIL=$(attr gts-email ops@femled.ai)
N_STAGING=$(attr gts-staging-n 60)
N_SHORTLIVED=$(attr gts-shortlived-n 1)
N_CADDY=$(attr gts-caddy-n 3)
N_PROD=$(attr gts-prod-n 8)
PAUSE=$(attr gts-pause 20)
PROJECT=$(md project/project-id)
MY_IP=$(md instance/network-interfaces/0/access-configs/0/external-ip)

STAGING_DIR=https://dv.acme-v02.test-api.pki.goog/directory
PROD_DIR=https://dv.acme-v02.api.pki.goog/directory
STAGING_EAB=https://preprod-publicca.googleapis.com/v1/projects/$PROJECT/locations/global/externalAccountKeys
PROD_EAB=https://publicca.googleapis.com/v1/projects/$PROJECT/locations/global/externalAccountKeys

STATE=/var/lib/gts-acme-test
WORK=/var/tmp/gts-acme-test
LOGDIR=/var/log/gts-acme-test
mkdir -p "$STATE" "$WORK" "$LOGDIR"
RESULTS=$LOGDIR/results.jsonl

say() { echo "GTSTEST $*"; }
emit() { echo "$1" >> "$RESULTS"; say "$1"; }
now_ms() { date +%s%3N; }
json_str() { printf '%s' "$1" | jq -Rs .; }

serve_static() {
  # Leave the last production certificate serving a page so the chain can be
  # inspected from phones. Survives reboots via the done marker below.
  local crt=$STATE/prod-last.crt key=$STATE/prod-last.key
  [ -s "$crt" ] && [ -s "$key" ] || return 0
  cat > "$STATE/static.Caddyfile" <<EOF
{
	admin off
	auto_https disable_redirects
}
https://$HOST {
	tls $crt $key
	header -Server
	respond "gts-acme-test: the last of the production iterations' certificates from Google Trust Services (page generated $(date -u +%FT%TZ)). See masseuse-video-tee/README.md, Certificate authority." 200
}
EOF
  pkill -x caddy 2>/dev/null; sleep 1
  nohup /usr/local/bin/caddy run --config "$STATE/static.Caddyfile" --adapter caddyfile \
    > "$LOGDIR/static-caddy.log" 2>&1 &
  say "static page up on https://$HOST with the last production certificate"
}

if [ -e "$STATE/done" ]; then
  say "already ran; restoring the static page only"
  serve_static
  exit 0
fi

say "start host=$HOST ip=$MY_IP project=$PROJECT staging=$N_STAGING shortlived=$N_SHORTLIVED caddy=$N_CADDY prod=$N_PROD pause=${PAUSE}s"
# A re-run (done marker removed, phase counts changed in metadata) must get
# :443 back from the static page left by the previous run.
pkill -x caddy 2>/dev/null; pkill -x lego-ku 2>/dev/null; pkill -x lego 2>/dev/null

# ---- tools -----------------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get -qq update >/dev/null 2>&1
apt-get -qq install -y jq curl ca-certificates openssl dnsutils >/dev/null 2>&1 || say "apt-get failed; continuing with what is installed"

# lego: the patched build (see the header) from gts-lego-url or already at
# /usr/local/bin/lego-ku; the upstream release only as a documented failure.
LEGO=/usr/local/bin/lego-ku
LEGO_URL=$(attr gts-lego-url "")
if [ ! -x "$LEGO" ] && [ -n "$LEGO_URL" ]; then
  curl -fsSL "$LEGO_URL" -o "$LEGO" && chmod +x "$LEGO" || say "patched lego download failed"
fi
if [ ! -x "$LEGO" ]; then
  say "WARNING: no patched lego (gts-lego-url unset, /usr/local/bin/lego-ku absent); using the upstream release, whose TLS-ALPN-01 challenge certificate GTS rejects"
  TAG=$(curl -fsSL https://api.github.com/repos/go-acme/lego/releases/latest | jq -r .tag_name)
  [ -n "$TAG" ] && [ "$TAG" != null ] || TAG=v5.4.1
  curl -fsSL "https://github.com/go-acme/lego/releases/download/$TAG/lego_${TAG}_linux_amd64.tar.gz" \
    | tar -xz -C /usr/local/bin lego || say "lego download failed"
  LEGO=/usr/local/bin/lego
fi
if ! command -v caddy >/dev/null; then
  curl -fsSL 'https://caddyserver.com/api/download?os=linux&arch=amd64' -o /usr/local/bin/caddy \
    && chmod +x /usr/local/bin/caddy || say "caddy download failed"
fi
say "tools lego=$("$LEGO" --version 2>/dev/null | head -1) ($LEGO) caddy=$(caddy version 2>/dev/null | head -1)"

# ---- wait for the A record -------------------------------------------------
# DNS only (grey cloud) is required: the TLS-ALPN-01 validator has to reach
# this VM, not Cloudflare's edge. Poll the zone's authoritative servers, not
# public resolvers: a resolver asked before the record exists caches the
# NXDOMAIN for the SOA minimum (1800 s here), and the CA's validators run
# their own resolvers anyway. Public resolvers are reported for information.
zone=$HOST; NS=""
while [ -n "$zone" ] && [ "$zone" != "${zone#*.}" ]; do
  zone=${zone#*.}
  NS=$(dig +short NS "$zone" @8.8.8.8 2>/dev/null | sed 's/\.$//' | sort)
  [ -n "$NS" ] && break
done
say "zone $zone, authoritative: $(echo $NS | tr '\n' ' ')"
doh() { curl -fsS -H 'accept: application/dns-json' "$1" 2>/dev/null | jq -r '.Answer[]? | select(.type==1) | .data' 2>/dev/null | head -1; }
auth_answer() {
  if [ -z "$NS" ]; then doh "https://dns.google/resolve?name=$HOST&type=A"; return; fi
  for ns in $NS; do dig +short +time=3 +tries=1 A "$HOST" "@$ns" 2>/dev/null | head -1; done | sort -u | tr '\n' ' ' | sed 's/ $//'
}
say "waiting for $HOST A $MY_IP at the authoritative servers"
t0=$(now_ms); n=0
while :; do
  ans=$(auth_answer)
  if [ "$ans" = "$MY_IP" ]; then break; fi
  n=$((n+1))
  if [ -n "$ans" ]; then say "authoritative answer [$ans] is not this VM (proxied record?); still waiting"
  elif [ $((n % 20)) -eq 1 ]; then say "no A record yet ($(date -u +%H:%M:%SZ))"; fi
  sleep 15
done
say "DNS ready at the authoritative servers after $(( ($(now_ms) - t0) / 1000 )) s; public resolvers: 8.8.8.8=[$(doh "https://dns.google/resolve?name=$HOST&type=A")] 1.1.1.1=[$(doh "https://cloudflare-dns.com/dns-query?name=$HOST&type=A")] (informational, they may still hold a cached NXDOMAIN)"
sleep 30

# ---- EAB -------------------------------------------------------------------
# externalAccountKeys.create with the VM service account's token. b64MacKey
# is a protobuf bytes field, so the JSON carries standard base64 of the
# base64url HMAC the ACME client wants; decode once. One EAB registers
# exactly one ACME account.
sa_token() { md instance/service-accounts/default/token | jq -r .access_token; }
mint_eab() { # $1 = endpoint; prints "kid hmac ms" or fails
  local t0 body kid raw dec
  t0=$(now_ms)
  body=$(curl -sS -X POST -H "Authorization: Bearer $(sa_token)" -H 'Content-Type: application/json' -d '{}' "$1") || return 1
  kid=$(printf '%s' "$body" | jq -r '.keyId // empty')
  raw=$(printf '%s' "$body" | jq -r '.b64MacKey // empty')
  if [ -z "$kid" ] || [ -z "$raw" ]; then
    say "EAB mint failed: $(printf '%s' "$body" | tr -d '\n' | cut -c1-400)"
    return 1
  fi
  dec=$(printf '%s' "$raw" | base64 -d 2>/dev/null | tr -d '\n')
  if printf '%s' "$dec" | grep -Eq '^[A-Za-z0-9_-]{32,}$'; then
    echo "$kid $dec $(( $(now_ms) - t0 )) decoded"
  else
    echo "$kid $raw $(( $(now_ms) - t0 )) raw"
  fi
}

cert_info() { # $1 = PEM file; prints JSON fragment
  local serial na issuer fp
  serial=$(openssl x509 -in "$1" -noout -serial 2>/dev/null | cut -d= -f2)
  na=$(openssl x509 -in "$1" -noout -enddate 2>/dev/null | cut -d= -f2)
  issuer=$(openssl x509 -in "$1" -noout -issuer -nameopt RFC2253 2>/dev/null | sed 's/^issuer=//')
  fp=$(openssl x509 -in "$1" -noout -fingerprint -sha256 2>/dev/null | cut -d= -f2)
  printf '"serial":%s,"notAfter":%s,"issuer":%s,"sha256":%s' \
    "$(json_str "$serial")" "$(json_str "$na")" "$(json_str "$issuer")" "$(json_str "$fp")"
}

classify() { # $1 = lego log; prints a short error class from the WARN/ERROR lines only
  local e
  e=$(grep -E 'level=(ERROR|WARN)|handshake error' "$1")
  if printf '%s' "$e" | grep -q 'rateLimited'; then echo rateLimited
  elif printf '%s' "$e" | grep -qE '\b429\b'; then echo http429
  elif printf '%s' "$e" | grep -qiE 'external account binding|externalAccountBinding|\(EAB\) key'; then echo eab
  elif printf '%s' "$e" | grep -qiE 'caa'; then echo caa
  elif printf '%s' "$e" | grep -q 'No response from'; then echo validation-connection
  elif printf '%s' "$e" | grep -qiE 'timeout|timed out|deadline'; then echo timeout
  elif printf '%s' "$e" | grep -qE 'urn:ietf:params:acme:error:[a-zA-Z]+'; then printf '%s' "$e" | grep -oE 'urn:ietf:params:acme:error:[a-zA-Z]+' | head -1 | sed 's/.*://'
  else echo other; fi
}

# ---- one lego issuance from a fresh account ----------------------------------
HMAC_FALLBACK_TRIED=0
lego_iter() { # $1 phase, $2 index, $3 acme dir, $4 eab endpoint, $5 extra lego args (may be empty)
  local phase=$1 i=$2 dir=$3 eab=$4 extra=$5
  local path=$WORK/$phase-$i log=$LOGDIR/$phase-$i.log
  rm -rf "$path"; mkdir -p "$path"
  local e kid hmac eab_ms mode t0 rc secs err cls info line
  if ! e=$(mint_eab "$eab"); then
    emit "{\"phase\":\"$phase\",\"i\":$i,\"ok\":false,\"stage\":\"eab\",\"error\":\"eab mint failed\",\"t\":\"$(date -u +%FT%TZ)\"}"
    return 1
  fi
  read -r kid hmac eab_ms mode <<< "$e"
  t0=$(now_ms)
  # lego 5: global flags before `run`, everything else after it.
  timeout 300 "$LEGO" --log.format text run --accept-tos --email "$EMAIL" --server "$dir" \
    --eab --eab.kid "$kid" --eab.hmac "$hmac" \
    --domains "$HOST" --tls --tls.address :443 --path "$path" \
    --cert.timeout 120 $extra > "$log" 2>&1
  rc=$?
  secs=$(awk "BEGIN{printf \"%.1f\", ($(now_ms) - $t0)/1000}")
  if [ $rc -eq 0 ] && [ -s "$path/certificates/$HOST.crt" ]; then
    info=$(cert_info "$path/certificates/$HOST.crt")
    line="{\"phase\":\"$phase\",\"i\":$i,\"ok\":true,\"seconds\":$secs,\"eab_ms\":$eab_ms,\"hmac\":\"$mode\",$info,\"t\":\"$(date -u +%FT%TZ)\"}"
    emit "$line"
    return 0
  fi
  cls=$(classify "$log")
  # First iteration only: if the EAB was rejected, the other HMAC encoding is
  # the likely cause; try once and record which one works.
  if [ "$cls" = eab ] && [ $HMAC_FALLBACK_TRIED -eq 0 ]; then
    HMAC_FALLBACK_TRIED=1
    say "EAB rejected with the $mode HMAC; retrying with the other encoding"
    local other
    if [ "$mode" = decoded ]; then other=$(printf '%s' "$hmac" | base64 -w0); else other=$(printf '%s' "$hmac" | base64 -d 2>/dev/null | tr -d '\n'); fi
    rm -rf "$path"; mkdir -p "$path"
    t0=$(now_ms)
    timeout 300 "$LEGO" --log.format text run --accept-tos --email "$EMAIL" --server "$dir" \
      --eab --eab.kid "$kid" --eab.hmac "$other" \
      --domains "$HOST" --tls --tls.address :443 --path "$path" \
      --cert.timeout 120 $extra > "$log" 2>&1
    rc=$?
    secs=$(awk "BEGIN{printf \"%.1f\", ($(now_ms) - $t0)/1000}")
    if [ $rc -eq 0 ] && [ -s "$path/certificates/$HOST.crt" ]; then
      info=$(cert_info "$path/certificates/$HOST.crt")
      emit "{\"phase\":\"$phase\",\"i\":$i,\"ok\":true,\"seconds\":$secs,\"eab_ms\":$eab_ms,\"hmac\":\"other-than-$mode\",$info,\"t\":\"$(date -u +%FT%TZ)\"}"
      return 0
    fi
    cls=$(classify "$log")
  fi
  err=$(grep -E 'error|Error|urn:ietf' "$log" | tail -3 | tr '\n' ' ' | cut -c1-600)
  emit "{\"phase\":\"$phase\",\"i\":$i,\"ok\":false,\"seconds\":$secs,\"eab_ms\":$eab_ms,\"hmac\":\"$mode\",\"rc\":$rc,\"class\":\"$cls\",\"error\":$(json_str "$err"),\"t\":\"$(date -u +%FT%TZ)\"}"
  return 1
}

# ---- one Caddy run with the enclave's issuer stanza -------------------------
caddy_iter() { # $1 index, $2 acme dir, $3 eab endpoint
  local i=$1 dir=$2 eab=$3
  local store=$WORK/caddy-$i log=$LOGDIR/caddy-$i.log cf=$WORK/caddy-$i.Caddyfile
  rm -rf "$store"; mkdir -p "$store"
  local e kid hmac eab_ms mode t0 pid deadline got secs info line
  if ! e=$(mint_eab "$eab"); then
    emit "{\"phase\":\"caddy\",\"i\":$i,\"ok\":false,\"stage\":\"eab\",\"error\":\"eab mint failed\",\"t\":\"$(date -u +%FT%TZ)\"}"
    return 1
  fi
  read -r kid hmac eab_ms mode <<< "$e"
  # Same shape as workload/tee/Caddyfile plus the eab
  # subdirective; storage is a fresh directory, as /run/tee/caddy is per boot.
  cat > "$cf" <<EOF
{
	admin off
	auto_https disable_redirects
	email $EMAIL
	acme_ca $dir
	storage file_system {
		root $store
	}
	log {
		output stdout
		format console
		level INFO
	}
}
$HOST {
	tls {
		issuer acme {
			dir $dir
			email $EMAIL
			eab $kid $hmac
			disable_http_challenge
		}
	}
	respond "gts-acme-test caddy $i" 200
}
EOF
  t0=$(now_ms)
  caddy run --config "$cf" --adapter caddyfile > "$log" 2>&1 &
  pid=$!
  deadline=$(( t0 + 240000 ))
  got=""
  while [ "$(now_ms)" -lt $deadline ]; do
    # Caddy's own log line is the signal; the served leaf is read afterwards.
    if grep -q 'certificate obtained successfully' "$log"; then got=yes; break; fi
    if grep -qE 'could not get certificate from issuer|"level":"error".*obtain' "$log"; then break; fi
    sleep 1
  done
  secs=$(awk "BEGIN{printf \"%.1f\", ($(now_ms) - $t0)/1000}")
  if [ -n "$got" ]; then
    sleep 1
    echo | timeout 5 openssl s_client -connect 127.0.0.1:443 -servername "$HOST" 2>/dev/null | openssl x509 > "$store/leaf.pem" 2>/dev/null
    info=$(cert_info "$store/leaf.pem")
    # Caddy's own obtain time, "obtaining certificate" to "obtained successfully".
    local ta tb csecs
    ta=$(grep -m1 'obtaining certificate' "$log" | awk '{print $1" "$2}')
    tb=$(grep -m1 'certificate obtained successfully' "$log" | awk '{print $1" "$2}')
    csecs=$(python3 -c "import sys,datetime as d; f=lambda s: d.datetime.strptime(s,'%Y/%m/%d %H:%M:%S.%f'); print(round((f(sys.argv[2])-f(sys.argv[1])).total_seconds(),1))" "$ta" "$tb" 2>/dev/null || echo null)
    line="{\"phase\":\"caddy\",\"i\":$i,\"ok\":true,\"seconds\":$secs,\"caddy_obtain_seconds\":$csecs,\"eab_ms\":$eab_ms,\"hmac\":\"$mode\",$info,\"t\":\"$(date -u +%FT%TZ)\"}"
    emit "$line"
  else
    local err
    err=$(grep -iE 'error|fail|429|rateLimited' "$log" | tail -3 | tr '\n' ' ' | cut -c1-600)
    emit "{\"phase\":\"caddy\",\"i\":$i,\"ok\":false,\"seconds\":$secs,\"eab_ms\":$eab_ms,\"hmac\":\"$mode\",\"error\":$(json_str "$err"),\"t\":\"$(date -u +%FT%TZ)\"}"
  fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 1
  [ -n "$got" ]
}

# ---- run -------------------------------------------------------------------
say "phase staging: $N_STAGING issuances from fresh accounts on $STAGING_DIR"
ok=0; bad=0
for i in $(seq 1 "$N_STAGING"); do
  if lego_iter staging "$i" "$STAGING_DIR" "$STAGING_EAB" ""; then ok=$((ok+1)); else bad=$((bad+1)); fi
  sleep "$PAUSE"
done
say "phase staging done ok=$ok failed=$bad"

if [ "$N_SHORTLIVED" -gt 0 ]; then
  say "phase shortlived: one staging issuance with notAfter = now + 7 d"
  NA=$(date -u -d '+7 days' +%FT%TZ)
  lego_iter shortlived 1 "$STAGING_DIR" "$STAGING_EAB" "--not-after $NA"
  sleep "$PAUSE"
fi

say "phase caddy: $N_CADDY fresh-storage runs of the enclave's issuer stanza on staging"
cok=0; cbad=0
for i in $(seq 1 "$N_CADDY"); do
  if caddy_iter "$i" "$STAGING_DIR" "$STAGING_EAB"; then cok=$((cok+1)); else cbad=$((cbad+1)); fi
  sleep "$PAUSE"
done
say "phase caddy done ok=$cok failed=$cbad"

say "phase production: $N_PROD issuances from fresh accounts on $PROD_DIR"
pok=0; pbad=0
for i in $(seq 1 "$N_PROD"); do
  if lego_iter production "$i" "$PROD_DIR" "$PROD_EAB" ""; then
    pok=$((pok+1))
    cp "$WORK/production-$i/certificates/$HOST.crt" "$STATE/prod-last.crt"
    cp "$WORK/production-$i/certificates/$HOST.key" "$STATE/prod-last.key"
  else pbad=$((pbad+1)); fi
  sleep "$PAUSE"
done
say "phase production done ok=$pok failed=$pbad"

# ---- summary ---------------------------------------------------------------
summary() {
  jq -s '
    group_by(.phase) | map({
      phase: .[0].phase,
      n: length,
      ok: map(select(.ok)) | length,
      failed: map(select(.ok|not)) | length,
      classes: (map(select(.ok|not) | .class // .stage // "unknown") | group_by(.) | map({(.[0]): length}) | add // {}),
      seconds: (map(select(.ok) | .seconds) | sort | if length==0 then null else ((length-1) as $last | {min: .[0], p50: .[(length*0.5|floor)], p95: .[([(length*0.95|floor), $last] | min)], max: .[$last]}) end),
      eab_ms: (map(select(.eab_ms) | .eab_ms) | sort | if length==0 then null else {p50: .[(length*0.5|floor)], max: .[-1]} end)
    })' "$RESULTS" 2>/dev/null | jq -c '.[]'
}
say "summary:"
summary | while read -r l; do say "SUMMARY $l"; done
say "hmac encodings used: $(jq -r '.hmac // empty' "$RESULTS" | sort | uniq -c | tr '\n' ' ')"
say "distinct serials: $(jq -r 'select(.ok) | .serial' "$RESULTS" | sort -u | wc -l) of $(jq -r 'select(.ok) | .serial' "$RESULTS" | wc -l) issuances"

touch "$STATE/done"
serve_static
say "done; results in $RESULTS, per-iteration logs in $LOGDIR"
