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
	"strings"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

// A stand-in slot: TLS (httptest's self-signed certificate), /attestation
// minting a token whose eat_nonce binds the caller's nonce, the evidence
// key and the server's own SPKI, and a JWKS publishing the signing key.
type fakeSlot struct {
	server   *httptest.Server
	jwks     *httptest.Server
	signer   *rsa.PrivateKey
	evidence ed25519.PublicKey
	claims   func(nonce string) jwt.MapClaims
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
					"image_digest":     "sha256:" + strings.Repeat("ab", 32),
					"image_reference":  "us-central1-docker.pkg.dev/prod-masseuse-video-tee/masseuse-video-tee/masseuse-video-tee@sha256:" + strings.Repeat("ab", 32),
					"env":              map[string]any{"TRAINER_URL": "https://trainer.example"},
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

func (fs *fakeSlot) config() *config {
	u, _ := url.Parse(fs.server.URL)
	return &config{
		origin:         "https://" + u.Host,
		jwksURL:        fs.jwks.URL,
		projectID:      "prod-masseuse-video-tee",
		imageRefPrefix: "us-central1-docker.pkg.dev/prod-masseuse-video-tee/masseuse-video-tee/",
		allowedDigests: []string{"sha256:" + strings.Repeat("ab", 32)},
		trainerURL:     "https://trainer.example",
		signerKeyIDs:   []string{"cafe"},
		insecureTLS:    true, // httptest's certificate
		timeout:        10 * time.Second,
	}
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
	if r.SignerKeyIDs[0] != "cafe" || r.ImageDigest != "sha256:"+strings.Repeat("ab", 32) {
		t.Fatalf("report: %+v", r)
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
		container["env"] = map[string]any{"TRAINER_URL": "https://elsewhere.example"}
		container["image_signatures"] = []map[string]any{{"key_id": "beef", "signature_algorithm": "ECDSA_P256_SHA256"}}
		c["submods"].(map[string]any)["nvidia_gpu"] = map[string]any{"cc_mode": "OFF", "gpus": []map[string]any{{"hwmodel": "GCP_NVIDIA_L4"}}}
	})
	r := runReport(t, fs.config())
	for _, want := range []string{"image.digest", "image.env.TRAINER_URL", "image.signature", "gpu.cc_mode", "gpu.hwmodel"} {
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
	for _, want := range []string{"nonce.fresh", "nonce.tls-spki"} {
		if !contains(failed(r), want) {
			t.Errorf("expected %s to fail; failed set %v", want, failed(r))
		}
	}
	if contains(failed(r), "nonce.evidence-key") {
		t.Errorf("evidence key nonce was present and should pass")
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
