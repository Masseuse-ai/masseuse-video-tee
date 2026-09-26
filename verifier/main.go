// Verifier for a masseuse.ai video slot (Confidential Space, a3-highgpu-1g).
//
// Anyone, from anywhere, can run this against a live slot and learn the
// same things the phone learns before it sends its camera
// (the masseuse.ai web app runs the same checks before it sends media), plus what the phone
// cannot check from inside a browser:
//
//  1. Fetches https://<slot>/attestation?nonce=<fresh random> over TLS and
//     keeps the leaf certificate the connection actually used.
//  2. Verifies the attestation JWT against Google's Confidential Space
//     signing keys (RS256, JWKS) and the standard claims: issuer, audience
//     (the slot's own origin), expiry, swname CONFIDENTIAL_SPACE, hwmodel
//     GCP_INTEL_TDX, secure boot, dbgstat, STABLE support attributes.
//  3. Checks the GPU claims: nvidia_gpu.cc_mode ON and an H100 in
//     confidential-computing mode, so the pose model runs on an
//     attestation-bound GPU, not just a TDX CPU.
//  4. Pins the workload: image digest in the allowed list, image reference
//     under the project's Artifact Registry, TRAINER_URL env as expected,
//     and (once signing is required) a cosign signature by the expected
//     KMS key in image_signatures[].
//  5. Checks the nonce bindings in eat_nonce: the caller's nonce (fresh
//     token, not a replay), sha256(evidence key) fetched from
//     /evidence-key (the key that signs each SDP answer's DTLS
//     fingerprint), and sha256(TLS SubjectPublicKeyInfo) of the leaf
//     certificate this very connection negotiated. The last one is what a
//     browser cannot do, and it is what proves the TLS endpoint you are
//     talking to terminates inside the enclave that minted the token.
//  6. Optionally shells out to `cosign verify --key` to confirm the
//     signature over the running digest independently of the launcher.
//
// Exit codes: 0 every check passed; 1 a check failed (report on stdout);
// 2 usage or configuration error.
//
// Model: auth-broker-tee/verifier (same libraries, same shape), minus the
// governance lineage that broker carries and plus the GPU and TLS checks
// a video enclave needs.
package main

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/MicahParks/keyfunc/v3"
	"github.com/golang-jwt/jwt/v5"
)

const (
	defaultJWKSURL  = "https://www.googleapis.com/service_accounts/v1/metadata/jwk/signer@confidentialspace-sign.iam.gserviceaccount.com"
	expectedIssuer  = "https://confidentialcomputing.googleapis.com"
	expectedSWName  = "CONFIDENTIAL_SPACE"
	expectedHWModel = "GCP_INTEL_TDX"
	expectedGPU     = "GCP_NVIDIA_H100"
	dbgstatProd     = "disabled-since-boot"
	dbgstatDebug    = "enabled"
)

type config struct {
	origin          string
	fetchBase       string
	jwksURL         string
	projectID       string
	imageRefPrefix  string
	allowedDigests  []string
	trainerURL      string
	signerKeyIDs    []string
	allowDebug      bool
	insecureTLS     bool
	cosignBin       string
	cosignPublicKey string
	timeout         time.Duration
	verbose         bool
}

type attestationDoc struct {
	Token        string   `json:"token"`
	Nonces       []string `json:"nonces"`
	TLSSpkiNonce *string  `json:"tlsSpkiNonce"`
	EvidenceKey  struct {
		Alg       string `json:"alg"`
		PublicKey string `json:"publicKey"`
		Nonce     string `json:"nonce"`
	} `json:"evidenceKey"`
}

type evidenceKeyDoc struct {
	Alg       string `json:"alg"`
	PublicKey string `json:"publicKey"`
	Nonce     string `json:"nonce"`
}

type csClaims struct {
	jwt.RegisteredClaims
	SWName            string         `json:"swname"`
	HWModel           string         `json:"hwmodel"`
	DbgStat           string         `json:"dbgstat"`
	SecBoot           bool           `json:"secboot"`
	EatNonce          any            `json:"eat_nonce"`
	Submods           map[string]any `json:"submods"`
	GoogleServiceAccs []string       `json:"google_service_accounts"`
}

type check struct {
	Name   string `json:"name"`
	OK     bool   `json:"ok"`
	Detail string `json:"detail,omitempty"`
}

type report struct {
	Origin       string   `json:"origin"`
	ImageDigest  string   `json:"imageDigest,omitempty"`
	InstanceID   string   `json:"instanceId,omitempty"`
	DbgStat      string   `json:"dbgstat,omitempty"`
	TLSSpki      string   `json:"tlsSpkiSha256,omitempty"`
	EvidenceKey  string   `json:"evidenceKey,omitempty"`
	SignerKeyIDs []string `json:"imageSignatureKeyIds,omitempty"`
	Checks       []check  `json:"checks"`
	Passed       bool     `json:"passed"`
}

func (r *report) add(name string, ok bool, detail string) {
	r.Checks = append(r.Checks, check{Name: name, OK: ok, Detail: detail})
}

func main() {
	cfg, err := parseFlags()
	if err != nil {
		fmt.Fprintln(os.Stderr, "usage:", err)
		os.Exit(2)
	}
	ctx, cancel := context.WithTimeout(context.Background(), cfg.timeout)
	defer cancel()

	r := &report{Origin: cfg.origin}
	if err := run(ctx, cfg, r); err != nil {
		r.add("verifier", false, err.Error())
	}
	r.Passed = true
	for _, c := range r.Checks {
		if !c.OK {
			r.Passed = false
		}
	}
	out, _ := json.MarshalIndent(r, "", "  ")
	fmt.Println(string(out))
	if !r.Passed {
		os.Exit(1)
	}
}

func parseFlags() (*config, error) {
	cfg := &config{}
	var digests, signers string
	flag.StringVar(&cfg.origin, "origin", "https://slot-0.tee.masseuse.ai", "the slot's origin (also the token audience)")
	flag.StringVar(&cfg.fetchBase, "fetch-base", "", "debug only: send the requests here instead of to -origin (an IAP tunnel to the producer's loopback port, http://127.0.0.1:18080); the audience is still -origin, and the TLS binding cannot pass because the connection is not to the slot")
	flag.StringVar(&cfg.jwksURL, "jwks-url", defaultJWKSURL, "Confidential Space attestation signer JWKS")
	flag.StringVar(&cfg.projectID, "project-id", "prod-masseuse-video-tee", "expected submods.gce.project_id")
	flag.StringVar(&cfg.imageRefPrefix, "image-ref-prefix", "us-central1-docker.pkg.dev/prod-masseuse-video-tee/masseuse-video-tee/", "expected prefix of submods.container.image_reference")
	flag.StringVar(&digests, "allowed-digests", "", "comma-separated sha256:... digests the slot may run (VERIFICATION.md publishes the list); empty prints the digest without pinning")
	flag.StringVar(&cfg.trainerURL, "trainer-url", "https://masseuse-trainer-125139120897.us-central1.run.app", "expected TRAINER_URL in the workload env (where readings go)")
	flag.StringVar(&signers, "signer-key-ids", "", "comma-separated hex sha256 fingerprints of accepted cosign signing keys (terraform output image_signer_fingerprint); empty skips the signature check")
	flag.BoolVar(&cfg.allowDebug, "allow-debug", false, "accept dbgstat=enabled and no STABLE attribute (a debug image; never for production verification)")
	flag.BoolVar(&cfg.insecureTLS, "insecure-tls", false, "do not verify the server certificate chain (Let's Encrypt staging during debug); the SPKI binding is still checked")
	flag.StringVar(&cfg.cosignBin, "cosign", "", "path to cosign; with -cosign-public-key, verifies the signature over the running digest independently")
	flag.StringVar(&cfg.cosignPublicKey, "cosign-public-key", "", "PEM file of the signing public key (terraform output image_signer_public_key_pem)")
	flag.DurationVar(&cfg.timeout, "timeout", 60*time.Second, "overall timeout")
	flag.BoolVar(&cfg.verbose, "v", false, "print the decoded claims")
	flag.Parse()

	cfg.origin = strings.TrimRight(cfg.origin, "/")
	if !strings.HasPrefix(cfg.origin, "https://") {
		return nil, errors.New("-origin must be https://")
	}
	cfg.fetchBase = strings.TrimRight(cfg.fetchBase, "/")
	if cfg.fetchBase != "" && !strings.HasPrefix(cfg.fetchBase, "http://") && !strings.HasPrefix(cfg.fetchBase, "https://") {
		return nil, errors.New("-fetch-base must be http:// or https://")
	}
	cfg.allowedDigests = splitTrim(digests)
	for _, d := range cfg.allowedDigests {
		if !strings.HasPrefix(d, "sha256:") || len(d) != 7+64 {
			return nil, fmt.Errorf("bad digest %q", d)
		}
	}
	cfg.signerKeyIDs = splitTrim(signers)
	if (cfg.cosignBin == "") != (cfg.cosignPublicKey == "") {
		return nil, errors.New("-cosign and -cosign-public-key go together")
	}
	return cfg, nil
}

func splitTrim(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func run(ctx context.Context, cfg *config, r *report) error {
	nonce, err := randomNonce(24)
	if err != nil {
		return err
	}

	base := cfg.origin
	if cfg.fetchBase != "" {
		base = cfg.fetchBase
	}
	tunnelled := strings.HasPrefix(base, "http://")

	client, tlsState := tlsClient(cfg.insecureTLS)
	doc, err := fetchAttestation(ctx, client, base, nonce)
	if err != nil {
		return fmt.Errorf("GET /attestation: %w", err)
	}
	if tunnelled {
		// A debug tunnel ends at the producer's loopback port, not at Caddy:
		// there is no certificate to bind, so this run cannot pass.
		r.add("tls.certificate", false, "fetched through "+base+", not from the slot's TLS endpoint: the TLS binding is unverified (debug only)")
	} else {
		leaf := tlsState()
		if leaf == nil {
			return errors.New("no TLS leaf certificate captured on the /attestation connection")
		}
		spki := sha256.Sum256(leaf.RawSubjectPublicKeyInfo)
		r.TLSSpki = b64url(spki[:])
		r.add("tls.certificate", true, fmt.Sprintf("%s (issuer %s, expires %s)", leaf.Subject.CommonName, leaf.Issuer.CommonName, leaf.NotAfter.UTC().Format(time.RFC3339)))
	}

	evidenceKey, err := fetchEvidenceKey(ctx, client, base)
	if err != nil {
		return fmt.Errorf("GET /evidence-key: %w", err)
	}
	r.EvidenceKey = evidenceKey.PublicKey
	keyRaw, err := base64.RawURLEncoding.DecodeString(strings.TrimRight(evidenceKey.PublicKey, "="))
	if err != nil || len(keyRaw) != 32 || evidenceKey.Alg != "Ed25519" {
		r.add("evidence.key", false, "not a raw 32-byte Ed25519 public key")
	} else {
		r.add("evidence.key", true, "Ed25519, 32 bytes")
	}
	keyHash := sha256.Sum256(keyRaw)
	keyNonce := b64url(keyHash[:])
	r.add("evidence.key.matches-attestation-document", evidenceKey.PublicKey == doc.EvidenceKey.PublicKey && keyNonce == evidenceKey.Nonce,
		"the key /evidence-key serves is the one /attestation describes, and its nonce is its sha256")

	claims, err := verifyJWT(ctx, cfg, doc.Token)
	if err != nil {
		r.add("jwt.signature", false, err.Error())
		return nil
	}
	r.add("jwt.signature", true, "RS256, signed by the Confidential Space attestation signer")
	r.ImageDigest = nestedString(claims.Submods, "container", "image_digest")
	r.InstanceID = nestedString(claims.Submods, "gce", "instance_id")
	r.DbgStat = claims.DbgStat
	if cfg.verbose {
		pretty, _ := json.MarshalIndent(claims, "", "  ")
		fmt.Fprintln(os.Stderr, string(pretty))
	}

	// Standard claims.
	r.add("jwt.issuer", claims.Issuer == expectedIssuer, claims.Issuer)
	r.add("jwt.audience", contains(claims.Audience, cfg.origin), strings.Join(claims.Audience, ","))
	r.add("cs.swname", claims.SWName == expectedSWName, claims.SWName)
	r.add("cs.hwmodel", claims.HWModel == expectedHWModel, claims.HWModel)
	r.add("cs.secboot", claims.SecBoot, fmt.Sprintf("%v", claims.SecBoot))
	r.add("cs.project", nestedString(claims.Submods, "gce", "project_id") == cfg.projectID, nestedString(claims.Submods, "gce", "project_id"))
	stable := containsAny(nestedAny(claims.Submods, "confidential_space", "support_attributes"), "STABLE")
	if cfg.allowDebug {
		r.add("cs.dbgstat", claims.DbgStat == dbgstatProd || claims.DbgStat == dbgstatDebug, claims.DbgStat+" (debug allowed)")
	} else {
		r.add("cs.dbgstat", claims.DbgStat == dbgstatProd, claims.DbgStat)
		r.add("cs.support_attributes.STABLE", stable, fmt.Sprintf("%v", nestedAny(claims.Submods, "confidential_space", "support_attributes")))
	}

	// GPU.
	r.add("gpu.cc_mode", nestedString(claims.Submods, "nvidia_gpu", "cc_mode") == "ON", nestedString(claims.Submods, "nvidia_gpu", "cc_mode"))
	gpuOK, gpuDetail := gpuModel(claims.Submods)
	r.add("gpu.hwmodel", gpuOK, gpuDetail)

	// Workload.
	if len(cfg.allowedDigests) > 0 {
		r.add("image.digest", contains(cfg.allowedDigests, r.ImageDigest), r.ImageDigest)
	} else {
		r.add("image.digest", r.ImageDigest != "", r.ImageDigest+" (not pinned: pass -allowed-digests)")
	}
	ref := nestedString(claims.Submods, "container", "image_reference")
	r.add("image.reference", strings.HasPrefix(ref, cfg.imageRefPrefix), ref)
	env, _ := nestedAny(claims.Submods, "container", "env").(map[string]any)
	trainer, _ := env["TRAINER_URL"].(string)
	r.add("image.env.TRAINER_URL", strings.TrimRight(trainer, "/") == strings.TrimRight(cfg.trainerURL, "/"), trainer)
	r.SignerKeyIDs = signatureKeyIDs(claims.Submods)
	if len(cfg.signerKeyIDs) > 0 {
		ok := false
		for _, id := range r.SignerKeyIDs {
			if contains(cfg.signerKeyIDs, id) {
				ok = true
			}
		}
		r.add("image.signature", ok, fmt.Sprintf("key ids in token: %v", r.SignerKeyIDs))
	}

	// Nonce bindings.
	r.add("nonce.fresh", nonceMatches(claims.EatNonce, nonce), "eat_nonce echoes this run's nonce")
	r.add("nonce.evidence-key", nonceMatches(claims.EatNonce, keyNonce), "eat_nonce contains sha256(evidence key)")
	if tunnelled {
		r.add("nonce.tls-spki", false, "not checkable through a tunnel (the document's tlsSpkiNonce, if any, is "+deref(doc.TLSSpkiNonce)+")")
	} else if doc.TLSSpkiNonce == nil {
		r.add("nonce.tls-spki", false, "attestation document carries no tlsSpkiNonce (certificate not issued yet?)")
	} else {
		r.add("nonce.tls-spki", nonceMatches(claims.EatNonce, r.TLSSpki) && *doc.TLSSpkiNonce == r.TLSSpki,
			"eat_nonce contains sha256(SPKI) of the certificate this connection negotiated")
	}

	// Independent cosign check.
	if cfg.cosignBin != "" && r.ImageDigest != "" {
		imageRef := strings.TrimSuffix(cfg.imageRefPrefix, "/") + "/masseuse-video-tee@" + r.ImageDigest
		if i := strings.Index(ref, "@"); i > 0 {
			imageRef = ref[:i] + "@" + r.ImageDigest
		}
		out, err := exec.CommandContext(ctx, cfg.cosignBin, "verify", "--key", cfg.cosignPublicKey, "--insecure-ignore-tlog", imageRef).CombinedOutput()
		r.add("cosign.verify", err == nil, truncate(strings.TrimSpace(string(out)), 300))
	}
	return nil
}

// tlsClient returns an HTTP client and a function that yields the leaf
// certificate of the most recent TLS connection it made.
func tlsClient(insecure bool) (*http.Client, func() *x509.Certificate) {
	var leaf *x509.Certificate
	transport := &http.Transport{
		TLSClientConfig: &tls.Config{
			InsecureSkipVerify: insecure, //nolint:gosec - debug staging certificates; the SPKI binding is still enforced
			MinVersion:         tls.VersionTLS12,
			VerifyConnection: func(cs tls.ConnectionState) error {
				if len(cs.PeerCertificates) > 0 {
					leaf = cs.PeerCertificates[0]
				}
				return nil
			},
		},
		DisableKeepAlives: false,
	}
	return &http.Client{Transport: transport}, func() *x509.Certificate { return leaf }
}

func fetchAttestation(ctx context.Context, client *http.Client, origin, nonce string) (*attestationDoc, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, origin+"/attestation?nonce="+nonce, nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("status %d: %s", resp.StatusCode, truncate(string(body), 200))
	}
	var doc attestationDoc
	if err := json.Unmarshal(body, &doc); err != nil {
		return nil, err
	}
	if doc.Token == "" {
		return nil, errors.New("no token in the attestation document")
	}
	return &doc, nil
}

func fetchEvidenceKey(ctx context.Context, client *http.Client, origin string) (*evidenceKeyDoc, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, origin+"/evidence-key", nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("status %d", resp.StatusCode)
	}
	var doc evidenceKeyDoc
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<16)).Decode(&doc); err != nil {
		return nil, err
	}
	return &doc, nil
}

func verifyJWT(ctx context.Context, cfg *config, token string) (*csClaims, error) {
	kf, err := keyfunc.NewDefaultCtx(ctx, []string{cfg.jwksURL})
	if err != nil {
		return nil, fmt.Errorf("load JWKS: %w", err)
	}
	claims := &csClaims{}
	parsed, err := jwt.NewParser(
		jwt.WithExpirationRequired(),
		jwt.WithValidMethods([]string{"RS256"}),
	).ParseWithClaims(token, claims, kf.Keyfunc)
	if err != nil {
		return nil, fmt.Errorf("parse JWT: %w", err)
	}
	if !parsed.Valid {
		return nil, errors.New("JWT invalid")
	}
	return claims, nil
}

func gpuModel(submods map[string]any) (bool, string) {
	gpus, _ := nestedAny(submods, "nvidia_gpu", "gpus").([]any)
	var models []string
	ok := false
	for _, g := range gpus {
		m, _ := g.(map[string]any)
		model, _ := m["hwmodel"].(string)
		models = append(models, model)
		if model == expectedGPU {
			ok = true
		}
	}
	return ok, strings.Join(models, ",")
}

func signatureKeyIDs(submods map[string]any) []string {
	sigs, _ := nestedAny(submods, "container", "image_signatures").([]any)
	var ids []string
	for _, s := range sigs {
		m, _ := s.(map[string]any)
		if id, _ := m["key_id"].(string); id != "" {
			alg, _ := m["signature_algorithm"].(string)
			ids = append(ids, id)
			_ = alg
		}
	}
	return ids
}

func randomNonce(n int) (string, error) {
	buf := make([]byte, n)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	return b64url(buf), nil
}

func b64url(b []byte) string { return base64.RawURLEncoding.EncodeToString(b) }

func nonceMatches(eatNonce any, expected string) bool {
	switch v := eatNonce.(type) {
	case string:
		return v == expected
	case []any:
		for _, e := range v {
			if s, ok := e.(string); ok && s == expected {
				return true
			}
		}
	}
	return false
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

func containsAny(v any, s string) bool {
	list, _ := v.([]any)
	for _, e := range list {
		if str, ok := e.(string); ok && str == s {
			return true
		}
	}
	return false
}

func nestedAny(m map[string]any, path ...string) any {
	cur := any(m)
	for _, p := range path {
		mm, ok := cur.(map[string]any)
		if !ok {
			return nil
		}
		cur = mm[p]
	}
	return cur
}

func nestedString(m map[string]any, path ...string) string {
	s, _ := nestedAny(m, path...).(string)
	return s
}

func deref(s *string) string {
	if s == nil {
		return "absent"
	}
	return *s
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
