package main

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"math/big"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

const (
	testVersion = "v0.4.0"
	testCommit  = "0123456789abcdef0123456789abcdef01234567"
)

var testDigest = "sha256:" + strings.Repeat("ab", 32)

// A stand-in slot: TLS (httptest's self-signed certificate), /attestation
// minting a token whose eat_nonce binds the caller's nonce, the evidence
// key and the server's own SPKI, and a JWKS publishing the signing key.
type fakeSlot struct {
	server   *httptest.Server
	jwks     *httptest.Server
	signer   *rsa.PrivateKey
	evidence ed25519.PublicKey
	claims   func(nonce string) jwt.MapClaims
	// A stand-in slsa-verifier (stubSLSAVerifier) and where its arguments land.
	slsaBin, slsaArgs string
}

func newFakeSlot(t *testing.T, mutate func(jwt.MapClaims)) *fakeSlot {
	t.Helper()
	signer, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	evidence, _, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	keyHash := sha256.Sum256(evidence)
	keyNonce := b64url(keyHash[:])
	fs := &fakeSlot{signer: signer, evidence: evidence}
	fs.slsaBin, fs.slsaArgs = stubSLSAVerifier(t)

	jwks := map[string]any{"keys": []map[string]any{{
		"kty": "RSA", "kid": "test-kid", "alg": "RS256", "use": "sig",
		"n": base64.RawURLEncoding.EncodeToString(signer.PublicKey.N.Bytes()),
		"e": base64.RawURLEncoding.EncodeToString(big.NewInt(int64(signer.PublicKey.E)).Bytes()),
	}}}
	fs.jwks = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(jwks)
	}))
	t.Cleanup(fs.jwks.Close)

	mux := http.NewServeMux()
	fs.server = httptest.NewUnstartedServer(mux)
	fs.server.StartTLS()
	t.Cleanup(fs.server.Close)
	origin := fs.server.URL
	spki := sha256.Sum256(fs.server.Certificate().RawSubjectPublicKeyInfo)
	spkiNonce := b64url(spki[:])

	fs.claims = func(nonce string) jwt.MapClaims {
		now := time.Now()
		c := jwt.MapClaims{
			"iss": expectedIssuer, "aud": origin, "iat": now.Unix(), "exp": now.Add(time.Hour).Unix(),
			"swname": expectedSWName, "hwmodel": expectedHWModel, "secboot": true, "dbgstat": dbgstatProd,
			"eat_nonce":               []string{keyNonce, spkiNonce, nonce},
			"google_service_accounts": []string{"masseuse-video-tee-vm@prod-masseuse-video-tee.iam.gserviceaccount.com"},
			"submods": map[string]any{
				"gce":                map[string]any{"project_id": "prod-masseuse-video-tee", "instance_id": "1"},
				"confidential_space": map[string]any{"support_attributes": []string{"LATEST", "STABLE", "USABLE"}},
				"nvidia_gpu":         map[string]any{"cc_mode": "ON", "gpus": []map[string]any{{"hwmodel": expectedGPU}}},
				"container": map[string]any{
					"image_digest":    testDigest,
					"image_reference": "us-central1-docker.pkg.dev/prod-masseuse-video-tee/masseuse-video-tee/masseuse-video-tee@" + testDigest,
					"env": bootEnv(map[string]any{
						"TEE_CAPTURE_BUCKET": "records.example",
						"TEE_IMAGE_VERSION":  testVersion,
						"TEE_IMAGE_COMMIT":   testCommit,
					}),
					"image_signatures": []map[string]any{{"key_id": "cafe", "signature_algorithm": "ECDSA_P256_SHA256"}},
				},
			},
		}
		if mutate != nil {
			mutate(c)
		}
		return c
	}
	mux.HandleFunc("/attestation", func(w http.ResponseWriter, r *http.Request) {
		nonce := r.URL.Query().Get("nonce")
		tok := jwt.NewWithClaims(jwt.SigningMethodRS256, fs.claims(nonce))
		tok.Header["kid"] = "test-kid"
		signed, err := tok.SignedString(fs.signer)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"token": signed, "nonces": []string{keyNonce, spkiNonce, nonce}, "tlsSpkiNonce": spkiNonce,
			"evidenceKey": map[string]any{"alg": "Ed25519", "publicKey": b64url(evidence), "nonce": keyNonce},
		})
	})
	mux.HandleFunc("/evidence-key", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"alg": "Ed25519", "publicKey": b64url(evidence), "nonce": keyNonce})
	})
	return fs
}

// bootEnv: what the deployment sets on every production slot (TRAINER_URL,
// the request seal, the trainer as the one invoker), plus `extra`.
func bootEnv(extra map[string]any) map[string]any {
	env := map[string]any{
		"TRAINER_URL":                     "https://trainer.example",
		"TEE_REQUEST_SEAL":                "v1",
		"TRAINER_INVOKER_SERVICE_ACCOUNT": defaultTrainerInvoker,
	}
	for k, v := range extra {
		env[k] = v
	}
	return env
}

func (fs *fakeSlot) config() *config {
	u, _ := url.Parse(fs.server.URL)
	return &config{
		origin:           "https://" + u.Host,
		jwksURL:          fs.jwks.URL,
		projectIDs:       []string{"prod-masseuse-video-tee"},
		imageRefPrefixes: []string{"us-central1-docker.pkg.dev/prod-masseuse-video-tee/masseuse-video-tee/"},
		trainerURL:       "https://trainer.example",
		captureBucket:    "records.example",
		requestSeal:      "v1",
		trainerInvokers:  invokersOf(defaultTrainerInvoker),
		signerKeyIDs:     []string{"cafe"},
		slsaVerifierBin:  fs.slsaBin,
		sourceURI:        defaultSourceURI,
		imageRepo:        defaultImageRepo,
		insecureTLS:      true, // httptest's certificate
		timeout:          10 * time.Second,
	}
}

// stubSLSAVerifier writes a stand-in slsa-verifier that records its
// arguments, then prints a SLSA v1 provenance naming the commit in
// $STUB_COMMIT (or fails when $STUB_FAIL is set). Returns the binary path
// and the file the arguments land in.
func stubSLSAVerifier(t *testing.T) (bin, argsFile string) {
	t.Helper()
	dir := t.TempDir()
	bin = filepath.Join(dir, "slsa-verifier")
	argsFile = filepath.Join(dir, "args")
	script := `#!/bin/sh
printf '%s\n' "$@" > "$STUB_ARGS_FILE"
if [ -n "$STUB_FAIL" ]; then
  echo "FAILED: SLSA verification failed: $STUB_FAIL" >&2
  exit 1
fi
echo "Verified build using builder https://github.com/slsa-framework/slsa-github-generator/.github/workflows/generator_container_slsa3.yml@refs/tags/v2.1.0 at commit $STUB_COMMIT" >&2
echo "PASSED: SLSA verification passed" >&2
cat <<EOF
{"_type":"https://in-toto.io/Statement/v1","predicateType":"https://slsa.dev/provenance/v1","subject":[{"name":"ghcr.io/masseuse-ai/masseuse-video-tee","digest":{"sha256":"` + strings.Repeat("ab", 32) + `"}}],"predicate":{"buildDefinition":{"buildType":"https://slsa-framework.github.io/github-actions-buildtypes/workflow/v1","externalParameters":{"workflow":{"ref":"refs/tags/v0.4.0","repository":"https://github.com/Masseuse-ai/masseuse-video-tee","path":".github/workflows/release.yml"}},"resolvedDependencies":[{"uri":"git+https://github.com/Masseuse-ai/masseuse-video-tee@refs/tags/v0.4.0","digest":{"gitCommit":"$STUB_COMMIT"}}]},"runDetails":{"builder":{"id":"https://github.com/slsa-framework/slsa-github-generator/.github/workflows/generator_container_slsa3.yml@refs/tags/v2.1.0"}}}}
EOF
`
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("STUB_ARGS_FILE", argsFile)
	t.Setenv("STUB_COMMIT", testCommit)
	t.Setenv("STUB_FAIL", "")
	return bin, argsFile
}

func stubArgs(t *testing.T, argsFile string) []string {
	t.Helper()
	b, err := os.ReadFile(argsFile)
	if err != nil {
		t.Fatalf("slsa-verifier stub was not run: %v", err)
	}
	return strings.Split(strings.TrimSpace(string(b)), "\n")
}

func checkNamed(t *testing.T, r *report, name string) check {
	t.Helper()
	for _, c := range r.Checks {
		if c.Name == name {
			return c
		}
	}
	t.Fatalf("no check named %s in %+v", name, r.Checks)
	return check{}
}

func runReport(t *testing.T, cfg *config) *report {
	t.Helper()
	r := &report{Origin: cfg.origin}
	if err := run(context.Background(), cfg, r); err != nil {
		t.Fatalf("run: %v", err)
	}
	r.Passed = true
	for _, c := range r.Checks {
		if !c.OK {
			r.Passed = false
		}
	}
	return r
}

func failed(r *report) []string {
	var names []string
	for _, c := range r.Checks {
		if !c.OK {
			names = append(names, c.Name)
		}
	}
	return names
}

func TestPassesAgainstAConformingSlot(t *testing.T) {
	fs := newFakeSlot(t, nil)
	r := runReport(t, fs.config())
	if !r.Passed {
		t.Fatalf("expected pass, failed: %v", failed(r))
	}
	if r.SignerKeyIDs[0] != "cafe" || r.ImageDigest != testDigest {
		t.Fatalf("report: %+v", r)
	}
	if r.Release == nil || r.Release.Version != testVersion || r.Release.Commit != testCommit {
		t.Fatalf("release stamp not reported: %+v", r.Release)
	}
	digest := checkNamed(t, r, "image.digest")
	if !digest.OK || !strings.Contains(digest.Detail, "not pinned to a list") {
		t.Fatalf("image.digest should report without pinning: %+v", digest)
	}
	rel := checkNamed(t, r, "image.release")
	if !rel.OK || rel.Detail != testVersion+" @ "+testCommit {
		t.Fatalf("image.release: %+v", rel)
	}
}

func TestPinsTheDigestOnlyWhenAListIsGiven(t *testing.T) {
	fs := newFakeSlot(t, nil)
	cfg := fs.config()
	cfg.allowedDigests = []string{"sha256:" + strings.Repeat("cd", 32)}
	r := runReport(t, cfg)
	if d := checkNamed(t, r, "image.digest"); d.OK {
		t.Fatalf("a list that lacks the running digest must fail: %+v", d)
	}
	cfg.allowedDigests = []string{testDigest}
	if r := runReport(t, cfg); !r.Passed {
		t.Fatalf("the listed digest should pass, failed: %v", failed(r))
	}
}

func TestHoldsTheReleaseStampToTheFlags(t *testing.T) {
	fs := newFakeSlot(t, nil)
	cases := []struct {
		name          string
		min, expect   string
		pass          bool
		detailHasText string
	}{
		{"exact match", "", "v0.4.0", true, "v0.4.0"},
		{"other release expected", "", "v0.4.1", false, "expected v0.4.1"},
		{"at the floor", "v0.4.0", "", true, ""},
		{"above the floor", "v0.3.9", "", true, ""},
		{"below the floor", "v0.5.0", "", false, "older than the minimum v0.5.0"},
		{"pre-release floor", "v0.4.0-rc.1", "", true, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := fs.config()
			cfg.minRelease, cfg.expectRelease = tc.min, tc.expect
			r := runReport(t, cfg)
			rel := checkNamed(t, r, "image.release")
			if rel.OK != tc.pass {
				t.Fatalf("image.release ok=%v, want %v (%s)", rel.OK, tc.pass, rel.Detail)
			}
			if !strings.Contains(rel.Detail, tc.detailHasText) {
				t.Fatalf("detail %q lacks %q", rel.Detail, tc.detailHasText)
			}
			if r.Passed != tc.pass {
				t.Fatalf("passed=%v, failed set %v", r.Passed, failed(r))
			}
		})
	}
}

func TestTheCaptureBucketMustBeTheExpectedOneOrNone(t *testing.T) {
	// A slot with no bucket keeps no record: it passes only for a verifier
	// told to expect none, and a slot naming a bucket fails that verifier.
	bare := newFakeSlot(t, func(c jwt.MapClaims) {
		c["submods"].(map[string]any)["container"].(map[string]any)["env"] = bootEnv(map[string]any{
			"TEE_IMAGE_VERSION": testVersion, "TEE_IMAGE_COMMIT": testCommit})
	})
	r := runReport(t, bare.config())
	if !contains(failed(r), "image.env.TEE_CAPTURE_BUCKET") {
		t.Fatalf("a slot without the expected bucket must fail; failed %v", failed(r))
	}
	if c := checkNamed(t, r, "image.env.TEE_CAPTURE_BUCKET"); !strings.Contains(c.Detail, "keeps no session record") {
		t.Fatalf("detail should say the slot keeps no record: %+v", c)
	}
	cfg := bare.config()
	cfg.captureBucket = ""
	if r := runReport(t, cfg); !r.Passed {
		t.Fatalf("expecting no record, a bare slot passes; failed %v", failed(r))
	}
	recording := newFakeSlot(t, nil)
	cfg = recording.config()
	cfg.captureBucket = ""
	if r := runReport(t, cfg); !contains(failed(r), "image.env.TEE_CAPTURE_BUCKET") {
		t.Fatalf("a recording slot fails a verifier expecting none; failed %v", failed(r))
	}
}

func TestAnUnstampedImageCannotPassTheProvenanceCheck(t *testing.T) {
	fs := newFakeSlot(t, func(c jwt.MapClaims) {
		c["submods"].(map[string]any)["container"].(map[string]any)["env"] = bootEnv(map[string]any{"TEE_CAPTURE_BUCKET": "records.example"})
	})
	r := runReport(t, fs.config())
	// Without a floor the release row reports it; nothing ties it to a
	// tag and a commit, so the provenance cannot, and the run fails.
	if r.Release != nil || strings.Join(failed(r), ",") != "provenance.source" {
		t.Fatalf("an unstamped image fails on its provenance alone; failed %v, release %+v", failed(r), r.Release)
	}
	if rel := checkNamed(t, r, "image.release"); !rel.OK || !strings.Contains(rel.Detail, "unstamped") {
		t.Fatalf("detail should say unstamped: %+v", rel)
	}
	if prov := checkNamed(t, r, "provenance.source"); !strings.Contains(prov.Detail, "no release stamp") {
		t.Fatalf("provenance.source: %+v", prov)
	}
	if _, err := os.Stat(fs.slsaArgs); err == nil {
		t.Fatalf("slsa-verifier should not run for an unstamped image")
	}
	for _, set := range []func(*config){
		func(c *config) { c.minRelease = "v0.4.0" },
		func(c *config) { c.expectRelease = "v0.4.0" },
	} {
		cfg := fs.config()
		set(cfg)
		r := runReport(t, cfg)
		if rel := checkNamed(t, r, "image.release"); rel.OK || !strings.Contains(rel.Detail, "no release stamp") {
			t.Fatalf("a floor or an expectation must refuse an unstamped image: %+v", rel)
		}
	}
}

func TestRefusesAMalformedReleaseStamp(t *testing.T) {
	fs := newFakeSlot(t, func(c jwt.MapClaims) {
		env := c["submods"].(map[string]any)["container"].(map[string]any)["env"].(map[string]any)
		env["TEE_IMAGE_VERSION"] = "main"
	})
	r := runReport(t, fs.config())
	if rel := checkNamed(t, r, "image.release"); rel.OK || !strings.Contains(rel.Detail, "not a release tag") {
		t.Fatalf("image.release: %+v", rel)
	}
}

func TestVerifiesProvenanceAgainstTheStamp(t *testing.T) {
	fs := newFakeSlot(t, nil)
	cfg := fs.config()
	r := runReport(t, cfg)
	if !r.Passed {
		t.Fatalf("expected pass, failed: %v", failed(r))
	}
	prov := checkNamed(t, r, "provenance.source")
	if !strings.Contains(prov.Detail, defaultSourceURI+"@v0.4.0") || !strings.Contains(prov.Detail, "commit "+testCommit) {
		t.Fatalf("provenance detail: %q", prov.Detail)
	}
	args := stubArgs(t, fs.slsaArgs)
	want := []string{"verify-image", defaultImageRepo + "@" + testDigest, "--source-uri", defaultSourceURI, "--source-tag", testVersion, "--print-provenance"}
	if strings.Join(args, " ") != strings.Join(want, " ") {
		t.Fatalf("slsa-verifier args\n got %v\nwant %v", args, want)
	}

	// The provenance names a different commit than the stamp: the image
	// was not built from the source the tag points at.
	t.Setenv("STUB_COMMIT", strings.Repeat("9", 40))
	r = runReport(t, cfg)
	if prov := checkNamed(t, r, "provenance.source"); prov.OK || !strings.Contains(prov.Detail, "the image stamp says "+testCommit) {
		t.Fatalf("a commit mismatch must fail: %+v", prov)
	}

	// slsa-verifier itself refuses.
	t.Setenv("STUB_COMMIT", testCommit)
	t.Setenv("STUB_FAIL", "expected source github.com/Masseuse-ai/masseuse-video-tee, got github.com/other/repo")
	r = runReport(t, cfg)
	if prov := checkNamed(t, r, "provenance.source"); prov.OK || !strings.Contains(prov.Detail, "FAILED") {
		t.Fatalf("a slsa-verifier failure must fail: %+v", prov)
	}

	// A provenance whose commit cannot be read is no proof of the commit.
	t.Setenv("STUB_FAIL", "")
	t.Setenv("STUB_COMMIT", "")
	r = runReport(t, cfg)
	if prov := checkNamed(t, r, "provenance.source"); prov.OK || !strings.Contains(prov.Detail, "names no source commit") {
		t.Fatalf("a provenance without a commit must fail: %+v", prov)
	}
}

func TestARunThatCannotCheckTheSignatureOrTheProvenanceDoesNotPass(t *testing.T) {
	fs := newFakeSlot(t, nil)
	cfg := fs.config()
	cfg.signerKeyIDs = nil
	r := runReport(t, cfg)
	if sig := checkNamed(t, r, "image.signature"); sig.OK || !strings.Contains(sig.Detail, "not checked") || r.Passed {
		t.Fatalf("no accepted key must fail the run: %+v", sig)
	}
	cfg = fs.config()
	cfg.slsaVerifierBin = filepath.Join(t.TempDir(), "no-such-slsa-verifier")
	r = runReport(t, cfg)
	if prov := checkNamed(t, r, "provenance.source"); prov.OK || !strings.Contains(prov.Detail, "not found") || r.Passed {
		t.Fatalf("a missing slsa-verifier must fail the run: %+v", prov)
	}
	cfg.slsaVerifierBin = ""
	r = runReport(t, cfg)
	if prov := checkNamed(t, r, "provenance.source"); prov.OK || !strings.Contains(prov.Detail, "not checked") {
		t.Fatalf("an empty -slsa-verifier must fail the run: %+v", prov)
	}
}

func TestHoldsTheRequestSealAndTheInvokerAndReportsTheRest(t *testing.T) {
	withEnv := func(set map[string]any, drop ...string) *fakeSlot {
		return newFakeSlot(t, func(c jwt.MapClaims) {
			env := c["submods"].(map[string]any)["container"].(map[string]any)["env"].(map[string]any)
			for k, v := range set {
				env[k] = v
			}
			for _, k := range drop {
				delete(env, k)
			}
		})
	}
	ok := withEnv(map[string]any{"TRAINER_INVOKER_SERVICE_ACCOUNT": " " + strings.ToUpper(defaultTrainerInvoker) + ",", "MODELS_BUCKET": "prod-masseuse-video-tee-models"})
	r := runReport(t, ok.config())
	if !r.Passed {
		t.Fatalf("the trainer alone, read as the enclave reads it, passes; failed %v", failed(r))
	}
	for _, name := range reportedEnv {
		if c := checkNamed(t, r, "image.env."+name); !c.OK || !strings.Contains(c.Detail, "reported, not checked") {
			t.Fatalf("%s should be reported: %+v", name, c)
		}
	}
	if c := checkNamed(t, r, "image.env.MODELS_BUCKET"); !strings.HasPrefix(c.Detail, "prod-masseuse-video-tee-models") {
		t.Fatalf("MODELS_BUCKET: %+v", c)
	}
	cases := []struct {
		slot *fakeSlot
		row  string
		text string
	}{
		{withEnv(nil, "TEE_REQUEST_SEAL"), "image.env.TEE_REQUEST_SEAL", "bare bearer"},
		{withEnv(map[string]any{"TEE_REQUEST_SEAL": "v2"}), "image.env.TEE_REQUEST_SEAL", "v2"},
		{withEnv(map[string]any{"TRAINER_INVOKER_SERVICE_ACCOUNT": defaultTrainerInvoker + ",someone@prod-masseuse-video-tee.iam.gserviceaccount.com"}),
			"image.env.TRAINER_INVOKER_SERVICE_ACCOUNT", "someone@"},
		{withEnv(nil, "TRAINER_INVOKER_SERVICE_ACCOUNT"), "image.env.TRAINER_INVOKER_SERVICE_ACCOUNT", "expected exactly"},
	}
	for _, tc := range cases {
		r := runReport(t, tc.slot.config())
		if c := checkNamed(t, r, tc.row); c.OK || !strings.Contains(c.Detail, tc.text) || strings.Join(failed(r), ",") != tc.row {
			t.Fatalf("%s should fail alone (%q): %+v, failed %v", tc.row, tc.text, c, failed(r))
		}
	}
}

func TestCompareRelease(t *testing.T) {
	cases := []struct {
		a, b string
		want int
	}{
		{"v0.4.0", "v0.4.0", 0},
		{"v0.4.0", "v0.4.1", -1},
		{"v0.10.0", "v0.9.9", 1},
		{"v1.0.0", "v0.99.99", 1},
		{"v0.4.0-rc.1", "v0.4.0", -1},
		{"v0.4.0", "v0.4.0-rc.1", 1},
		{"v0.4.0-rc.1", "v0.4.0-rc.2", -1},
		{"v0.4.0+build.7", "v0.4.0", 0},
	}
	for _, tc := range cases {
		if got := compareRelease(tc.a, tc.b); got != tc.want {
			t.Errorf("compareRelease(%s, %s) = %d, want %d", tc.a, tc.b, got, tc.want)
		}
	}
	for _, bad := range []string{"", "0.4.0", "v0.4", "v0.4.0.1", "v0.4.0-", "main", "va.b.c", "sha256:abc"} {
		if validRelease(bad) {
			t.Errorf("%q should not be a release tag", bad)
		}
	}
	for _, good := range []string{"v0.4.0", "v10.0.3", "v0.4.0-rc.1", "v0.4.0+build", "v0.4.0-beta+exp.sha.5114f85"} {
		if !validRelease(good) {
			t.Errorf("%q should be a release tag", good)
		}
	}
}

func TestProvenanceSourceReadsBothPredicateShapes(t *testing.T) {
	v1 := `PASSED: noise on the same stream
{"predicateType":"https://slsa.dev/provenance/v1","predicate":{"buildDefinition":{"resolvedDependencies":[{"uri":"git+https://github.com/Masseuse-ai/masseuse-video-tee@refs/tags/v0.4.0","digest":{"gitCommit":"` + testCommit + `"}}]}}}`
	if c, ref := provenanceSource([]byte(v1)); c != testCommit || ref != "v0.4.0" {
		t.Fatalf("v1: commit %q ref %q", c, ref)
	}
	v02 := `{"predicateType":"https://slsa.dev/provenance/v0.2","predicate":{"invocation":{"configSource":{"uri":"git+https://github.com/Masseuse-ai/masseuse-video-tee@refs/tags/v0.3.1","digest":{"sha1":"` + testCommit + `"}}},"materials":[{"uri":"git+https://github.com/Masseuse-ai/masseuse-video-tee@refs/tags/v0.3.1","digest":{"sha1":"` + testCommit + `"}}]}}`
	if c, ref := provenanceSource([]byte(v02)); c != testCommit || ref != "v0.3.1" {
		t.Fatalf("v0.2: commit %q ref %q", c, ref)
	}
	if c, ref := provenanceSource([]byte("no json here")); c != "" || ref != "" {
		t.Fatalf("garbage: commit %q ref %q", c, ref)
	}
}

func TestRefusesDebugUnlessAllowed(t *testing.T) {
	fs := newFakeSlot(t, func(c jwt.MapClaims) {
		c["dbgstat"] = dbgstatDebug
		c["submods"].(map[string]any)["confidential_space"] = map[string]any{}
	})
	r := runReport(t, fs.config())
	if r.Passed || !contains(failed(r), "cs.dbgstat") || !contains(failed(r), "cs.support_attributes.STABLE") {
		t.Fatalf("expected dbgstat and STABLE failures, got %v", failed(r))
	}
	cfg := fs.config()
	cfg.allowDebug = true
	if r := runReport(t, cfg); !r.Passed {
		t.Fatalf("allow-debug should pass, failed: %v", failed(r))
	}
}

func TestRefusesTheWrongImageGpuTrainerAndSigner(t *testing.T) {
	fs := newFakeSlot(t, func(c jwt.MapClaims) {
		container := c["submods"].(map[string]any)["container"].(map[string]any)
		container["image_digest"] = "sha256:" + strings.Repeat("cd", 32)
		container["env"] = map[string]any{"TRAINER_URL": "https://elsewhere.example", "TEE_CAPTURE_BUCKET": "elsewhere-bucket"}
		container["image_signatures"] = []map[string]any{{"key_id": "beef", "signature_algorithm": "ECDSA_P256_SHA256"}}
		c["submods"].(map[string]any)["nvidia_gpu"] = map[string]any{"cc_mode": "OFF", "gpus": []map[string]any{{"hwmodel": "GCP_NVIDIA_L4"}}}
	})
	cfg := fs.config()
	cfg.allowedDigests = []string{testDigest}
	r := runReport(t, cfg)
	for _, want := range []string{"image.digest", "image.env.TRAINER_URL", "image.env.TEE_CAPTURE_BUCKET", "image.signature", "gpu.cc_mode", "gpu.hwmodel"} {
		if !contains(failed(r), want) {
			t.Errorf("expected %s to fail; failed set %v", want, failed(r))
		}
	}
}

func TestRefusesAStaleOrUnboundToken(t *testing.T) {
	// A token whose eat_nonce lacks the caller's nonce and the TLS SPKI is
	// a replay, or a proxy in front of the enclave.
	fs := newFakeSlot(t, func(c jwt.MapClaims) {
		c["eat_nonce"] = []string{c["eat_nonce"].([]string)[0]}
	})
	r := runReport(t, fs.config())
	for _, want := range []string{"nonce.fresh", "nonce.tls-spki", "nonce.exact"} {
		if !contains(failed(r), want) {
			t.Errorf("expected %s to fail; failed set %v", want, failed(r))
		}
	}
	if contains(failed(r), "nonce.evidence-key") {
		t.Errorf("evidence key nonce was present and should pass")
	}
}

func TestRefusesATokenThatCarriesAnotherNonce(t *testing.T) {
	// Every binding this run looks for is there, plus the hash of somebody
	// else's key: not a token the enclave minted for this run alone.
	stranger := sha256.Sum256([]byte("somebody else's key"))
	fs := newFakeSlot(t, func(c jwt.MapClaims) {
		c["eat_nonce"] = append(c["eat_nonce"].([]string), b64url(stranger[:]))
	})
	r := runReport(t, fs.config())
	if got := failed(r); len(got) != 1 || got[0] != "nonce.exact" {
		t.Fatalf("expected nonce.exact alone to fail, failed set %v", got)
	}
	if detail := checkNamed(t, r, "nonce.exact").Detail; !strings.Contains(detail, b64url(stranger[:])) {
		t.Fatalf("nonce.exact should list the token's nonces: %s", detail)
	}
}

func TestRefusesAForgedSignature(t *testing.T) {
	fs := newFakeSlot(t, nil)
	other, _ := rsa.GenerateKey(rand.Reader, 2048)
	fs.signer = other // the JWKS still publishes the original key
	r := runReport(t, fs.config())
	if !contains(failed(r), "jwt.signature") {
		t.Fatalf("expected jwt.signature to fail, got %v", failed(r))
	}
}
