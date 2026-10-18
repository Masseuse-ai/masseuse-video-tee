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
//  4. Pins the workload: a signature by the release KMS key in
//     image_signatures[] (the launcher verified it before starting the
//     image; the release workflow signs with the key, and so can whoever
//     the enclave project's IAM lets, which is why step 6 is not optional),
//     the image reference under the project's Artifact Registry, the
//     environment the deployment sets at boot as expected (TRAINER_URL,
//     where readings go; TEE_CAPTURE_BUCKET, the one bucket a leased
//     session's record may be written to; TEE_REQUEST_SEAL v1, the phone's
//     requests sealed and a bare bearer refused; exactly the trainer as
//     TRAINER_INVOKER_SERVICE_ACCOUNT, the one identity the control routes
//     take a token from), the rest of what it may set reported, and the
//     release stamp the workflow baked into the image
//     (TEE_IMAGE_VERSION, TEE_IMAGE_COMMIT in the attested environment):
//     which release is running, held to -expect-release / -min-release
//     when given. The digest is reported, and pinned only if a list is
//     passed with -allowed-digests: no list of digests lives anywhere,
//     since one kept in a repository is always a release behind the image.
//  5. Checks the nonce bindings in eat_nonce: the caller's nonce (fresh
//     token, not a replay), sha256(evidence key) fetched from
//     /evidence-key (the key that signs each SDP answer's DTLS
//     fingerprint), and sha256(TLS SubjectPublicKeyInfo) of the leaf
//     certificate this very connection negotiated. The last one is what a
//     browser cannot do, and it is what proves the TLS endpoint you are
//     talking to terminates inside the enclave that minted the token.
//     eat_nonce must hold those three and nothing else: the enclave
//     appends one caller's nonce and refuses one shaped like its own key
//     hashes, so no other party's key can sit in the token beside them.
//  6. Shells out to `slsa-verifier verify-image` for the digest the token
//     names, in the public registry, against the source repository at the
//     release the token names: the SLSA provenance proves which commit of
//     the public source produced the running image, and its commit must be
//     the one the image stamp says. A run that cannot make this check (no
//     slsa-verifier, an unstamped image, a provenance naming no commit),
//     or the signature check (no accepted key), does not pass.
//  7. Optionally shells out to `cosign verify --key` to confirm the
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
	"bytes"
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
	"io/fs"
	"net/http"
	"os"
	"os/exec"
	"slices"
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
	// Where the enclave image is built and published: this repository and
	// its registry, from v0.12.4 on (VERIFY.md, "Release history"). For an
	// image released up to v0.12.3, built at the project's previous home,
	// -source-uri and -image-repo name that home.
	defaultSourceURI = "github.com/Masseuse-ai/masseuse-video-tee"
	defaultImageRepo = "ghcr.io/masseuse-ai/masseuse-video-tee"
	// The release key's fingerprint (VERIFY.md, terraform output
	// image_signer_fingerprint), and the trainer's runtime account: the
	// one identity the slot's control routes should take a token from.
	defaultSignerKeyID    = "cfb085b950e93abb8332cede62fa50df662ef9aebb1533b2ae0bf1403ea4f811"
	defaultTrainerInvoker = "masseuse-trainer@prod-femled-couple-router.iam.gserviceaccount.com"
)

// The variables the launch policy lets the deployment set at boot
// (Dockerfile.tee allow_env_override) that this tool reports and does
// not hold to a value: what each can and cannot change is in VERIFY.md.
var reportedEnv = []string{"MODELS_BUCKET", "WIF_AUDIENCE", "ACME_DIRECTORY_URL", "TEE_ALLOWED_ORIGINS"}

type config struct {
	origin           string
	fetchBase        string
	jwksURL          string
	policyURL        string
	projectIDs       []string
	imageRefPrefixes []string
	allowedDigests   []string
	trainerURL       string
	captureBucket    string
	requestSeal      string
	trainerInvokers  []string
	signerKeyIDs     []string
	minRelease       string
	expectRelease    string
	slsaVerifierBin  string
	sourceURI        string
	imageRepo        string
	allowDebug       bool
	insecureTLS      bool
	cosignBin        string
	cosignPublicKey  string
	timeout          time.Duration
	verbose          bool
}

// release is the stamp the release workflow bakes into the image and the
// launcher attests with the rest of the container environment.
type release struct {
	Version string `json:"version"`
	Commit  string `json:"commit"`
}

// releaseOf reads the stamp off the attested environment; nil for an image
// built before the stamp existed.
func releaseOf(env map[string]any) *release {
	version, _ := env["TEE_IMAGE_VERSION"].(string)
	if version == "" {
		return nil
	}
	commit, _ := env["TEE_IMAGE_COMMIT"].(string)
	return &release{Version: version, Commit: commit}
}

// releaseNumbers parses "vMAJOR.MINOR.PATCH[-pre][+build]"; ok is false for
// anything else.
func releaseNumbers(tag string) (nums [3]int, pre string, ok bool) {
	if !strings.HasPrefix(tag, "v") {
		return nums, "", false
	}
	rest := tag[1:]
	if i := strings.IndexByte(rest, '+'); i >= 0 {
		rest = rest[:i]
	}
	if i := strings.IndexByte(rest, '-'); i >= 0 {
		rest, pre = rest[:i], rest[i+1:]
		if pre == "" {
			return nums, "", false
		}
	}
	parts := strings.Split(rest, ".")
	if len(parts) != 3 {
		return nums, "", false
	}
	for i, p := range parts {
		if p == "" || len(p) > 9 {
			return nums, "", false
		}
		n := 0
		for _, c := range p {
			if c < '0' || c > '9' {
				return nums, "", false
			}
			n = n*10 + int(c-'0')
		}
		nums[i] = n
	}
	return nums, pre, true
}

func validRelease(tag string) bool {
	_, _, ok := releaseNumbers(tag)
	return ok
}

// compareRelease orders two release tags: -1, 0 or +1. A pre-release
// precedes its release; two pre-releases of one version compare as strings.
func compareRelease(a, b string) int {
	an, ap, _ := releaseNumbers(a)
	bn, bp, _ := releaseNumbers(b)
	for i := range an {
		if an[i] != bn[i] {
			if an[i] < bn[i] {
				return -1
			}
			return 1
		}
	}
	switch {
	case ap == bp:
		return 0
	case ap == "":
		return 1
	case bp == "":
		return -1
	case ap < bp:
		return -1
	default:
		return 1
	}
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
	Policy       string   `json:"policy,omitempty"`
	ImageDigest  string   `json:"imageDigest,omitempty"`
	Release      *release `json:"release,omitempty"`
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

	r := &report{Origin: cfg.origin, Policy: cfg.policyURL}
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
	var digests, signers, invokers, projects, prefixes string
	flag.StringVar(&cfg.origin, "origin", "https://slot-0.tee.masseuse.ai", "the slot's origin (also the token audience)")
	flag.StringVar(&cfg.fetchBase, "fetch-base", "", "debug only: send the requests here instead of to -origin (an IAP tunnel to the producer's loopback port, http://127.0.0.1:18080); the audience is still -origin, and the TLS binding cannot pass because the connection is not to the slot")
	flag.StringVar(&cfg.jwksURL, "jwks-url", defaultJWKSURL, "Confidential Space attestation signer JWKS")
	flag.StringVar(&cfg.policyURL, "policy-url", "", "an https:// URL or a file holding the values a service holds its slots to (a JSON document of them, or one carrying them as its \"verify\" member); each replaces the default of the flag of the same meaning, and a flag given on the command line wins (VERIFY.md, \"Which values to verify against\")")
	flag.StringVar(&projects, "project-id", "prod-masseuse-video-tee", "comma-separated projects submods.gce.project_id may be")
	flag.StringVar(&prefixes, "image-ref-prefix", "us-central1-docker.pkg.dev/prod-masseuse-video-tee/masseuse-video-tee/", "comma-separated prefixes one of which submods.container.image_reference must start with")
	flag.StringVar(&digests, "allowed-digests", "", "comma-separated sha256:... digests to pin the slot to; empty (the norm) reports the digest and relies on the signature and the release stamp")
	flag.StringVar(&cfg.trainerURL, "trainer-url", "https://masseuse-trainer-125139120897.us-central1.run.app", "expected TRAINER_URL in the workload env (where readings go)")
	flag.StringVar(&cfg.captureBucket, "capture-bucket", "masseuse-ai-prod", "expected TEE_CAPTURE_BUCKET in the workload env (the bucket a leased session's record is written to, under the prefix the trainer's lease names); \"\" requires the slot to keep no record at all")
	flag.StringVar(&cfg.requestSeal, "request-seal", "v1", "expected TEE_REQUEST_SEAL in the workload env (requests under the phone's capability sealed to a key the enclave attests, and a bare bearer refused); \"\" expects it unset")
	flag.StringVar(&invokers, "trainer-invoker", defaultTrainerInvoker, "comma-separated service accounts TRAINER_INVOKER_SERVICE_ACCOUNT in the workload env must name, exactly: the only identities the slot's control routes take a token from")
	flag.StringVar(&signers, "signer-key-ids", defaultSignerKeyID, "comma-separated hex sha256 fingerprints of accepted cosign signing keys (terraform output image_signer_fingerprint, or VERIFY.md); the run cannot pass without one")
	flag.StringVar(&cfg.minRelease, "min-release", "", "the lowest release (vX.Y.Z) the image may be, held against its attested TEE_IMAGE_VERSION; empty accepts any, including images built before the stamp")
	flag.StringVar(&cfg.expectRelease, "expect-release", "", "the exact release (vX.Y.Z) the image must be stamped with (a roll's check that the new image is what booted)")
	flag.StringVar(&cfg.slsaVerifierBin, "slsa-verifier", "slsa-verifier", "slsa-verifier (a path, or found on PATH): verifies the SLSA provenance of the running digest in -image-repo against -source-uri at the release the token names, and that the provenance's commit is the image's TEE_IMAGE_COMMIT; the run cannot pass without it")
	flag.StringVar(&cfg.sourceURI, "source-uri", defaultSourceURI, "the source repository the provenance must name")
	flag.StringVar(&cfg.imageRepo, "image-repo", defaultImageRepo, "the public registry holding the same digest with its provenance")
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
	cfg.trainerInvokers = invokersOf(invokers)
	cfg.projectIDs = splitTrim(projects)
	cfg.imageRefPrefixes = splitTrim(prefixes)
	if cfg.policyURL != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()
		policy, err := readPolicy(ctx, &http.Client{Timeout: 20 * time.Second}, cfg.policyURL)
		if err != nil {
			return nil, fmt.Errorf("-policy-url: %w", err)
		}
		set := map[string]bool{}
		flag.Visit(func(f *flag.Flag) { set[f.Name] = true })
		applyPolicy(cfg, policy, set)
	}
	if len(cfg.projectIDs) == 0 || len(cfg.imageRefPrefixes) == 0 {
		return nil, errors.New("-project-id and -image-ref-prefix name where the slot must run from; neither may be empty")
	}
	if cfg.minRelease != "" && !validRelease(cfg.minRelease) {
		return nil, fmt.Errorf("-min-release %q is not a release tag (vX.Y.Z)", cfg.minRelease)
	}
	if cfg.expectRelease != "" && !validRelease(cfg.expectRelease) {
		return nil, fmt.Errorf("-expect-release %q is not a release tag (vX.Y.Z)", cfg.expectRelease)
	}
	if cfg.sourceURI == "" || cfg.imageRepo == "" {
		return nil, errors.New("-source-uri and -image-repo name where the provenance is checked; neither may be empty")
	}
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
	r.add("cs.project", contains(cfg.projectIDs, nestedString(claims.Submods, "gce", "project_id")), nestedString(claims.Submods, "gce", "project_id"))
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

	// Workload. The digest is the identity the token names; what pins it is
	// the signature (the release key, usable only by the release workflow)
	// and the release stamp, not a list.
	if len(cfg.allowedDigests) > 0 {
		r.add("image.digest", contains(cfg.allowedDigests, r.ImageDigest), r.ImageDigest)
	} else {
		r.add("image.digest", r.ImageDigest != "", r.ImageDigest+" (reported, not pinned to a list: the signature and the release stamp identify the image)")
	}
	ref := nestedString(claims.Submods, "container", "image_reference")
	r.add("image.reference", slices.ContainsFunc(cfg.imageRefPrefixes, func(p string) bool { return strings.HasPrefix(ref, p) }), ref)
	env, _ := nestedAny(claims.Submods, "container", "env").(map[string]any)
	trainer, _ := env["TRAINER_URL"].(string)
	r.add("image.env.TRAINER_URL", strings.TrimRight(trainer, "/") == strings.TrimRight(cfg.trainerURL, "/"), trainer)
	// Where a leased session's record goes (workload/producer/record.py):
	// the one bucket, or none. The trainer holds the same expectation
	// before it leases (expectedCaptureBucket in its policy); this is the
	// outsider's copy of that check.
	capture, _ := env["TEE_CAPTURE_BUCKET"].(string)
	captureDetail := capture
	if capture == "" {
		captureDetail = "(unset: the slot keeps no session record)"
	}
	r.add("image.env.TEE_CAPTURE_BUCKET", capture == cfg.captureBucket, captureDetail)
	for _, c := range bootEnvChecks(cfg, env) {
		r.add(c.Name, c.OK, c.Detail)
	}
	r.Release = releaseOf(env)
	relOK, relDetail := releaseCheck(cfg, r.Release)
	r.add("image.release", relOK, relDetail)
	r.SignerKeyIDs = signatureKeyIDs(claims.Submods)
	sigOK, sigDetail := signatureCheck(cfg.signerKeyIDs, r.SignerKeyIDs)
	r.add("image.signature", sigOK, sigDetail)

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
	want := []string{keyNonce, nonce}
	if doc.TLSSpkiNonce != nil {
		want = append(want, *doc.TLSSpkiNonce)
	}
	got := eatNonces(claims.EatNonce)
	r.add("nonce.exact", sameNonces(got, want),
		"eat_nonce holds the evidence key, the TLS key and this run's nonce, and nothing else: ["+strings.Join(got, ", ")+"]")

	// Provenance: the public copy of the running digest, its SLSA
	// provenance, the source repository at the release the token names.
	// Not optional: without it nothing outside the enclave's own project
	// says which build of the public source is running.
	switch {
	case r.ImageDigest == "":
		r.add("provenance.source", false, "the token names no image digest")
	case cfg.slsaVerifierBin == "":
		r.add("provenance.source", false, "not checked: -slsa-verifier is empty")
	default:
		provOK, provDetail := provenanceCheck(ctx, cfg, r.ImageDigest, r.Release)
		r.add("provenance.source", provOK, provDetail)
	}

	// Independent cosign check.
	if cfg.cosignBin != "" && r.ImageDigest != "" {
		imageRef := strings.TrimSuffix(cfg.imageRefPrefixes[0], "/") + "/masseuse-video-tee@" + r.ImageDigest
		if i := strings.Index(ref, "@"); i > 0 {
			imageRef = ref[:i] + "@" + r.ImageDigest
		}
		out, err := exec.CommandContext(ctx, cfg.cosignBin, "verify", "--key", cfg.cosignPublicKey, "--insecure-ignore-tlog", imageRef).CombinedOutput()
		r.add("cosign.verify", err == nil, truncate(strings.TrimSpace(string(out)), 300))
	}
	return nil
}

// releaseCheck holds the image's release stamp to -expect-release and
// -min-release. Without either, the stamp is reported and an unstamped
// image (built before the stamp existed) passes, saying so.
func releaseCheck(cfg *config, rel *release) (bool, string) {
	if rel == nil {
		if cfg.expectRelease != "" || cfg.minRelease != "" {
			return false, "image carries no release stamp (TEE_IMAGE_VERSION), built before releases were stamped"
		}
		return true, "unstamped (built before releases were stamped); the signature alone identifies the build"
	}
	detail := rel.Version
	if rel.Commit != "" {
		detail += " @ " + rel.Commit
	}
	if !validRelease(rel.Version) {
		return false, detail + " (not a release tag)"
	}
	if cfg.expectRelease != "" && rel.Version != cfg.expectRelease {
		return false, detail + " (expected " + cfg.expectRelease + ")"
	}
	if cfg.minRelease != "" && compareRelease(rel.Version, cfg.minRelease) < 0 {
		return false, detail + " (older than the minimum " + cfg.minRelease + ")"
	}
	return true, detail
}

// signatureCheck: one of the token's image_signatures key ids is an
// accepted key. With no accepted key nothing is checked, and that fails.
func signatureCheck(accepted, inToken []string) (bool, string) {
	if len(accepted) == 0 {
		return false, "not checked: -signer-key-ids names no key"
	}
	for _, id := range inToken {
		if contains(accepted, id) {
			return true, fmt.Sprintf("key ids in token: %v", inToken)
		}
	}
	return false, fmt.Sprintf("key ids in token: %v, none of %v", inToken, accepted)
}

// bootEnvChecks: the variables of the attested environment the deployment
// sets at boot that decide who may drive the slot and how the phone's
// requests reach it, held to their values; and the others it may set,
// reported.
func bootEnvChecks(cfg *config, env map[string]any) []check {
	seal, _ := env["TEE_REQUEST_SEAL"].(string)
	sealDetail := seal
	if seal == "" {
		sealDetail = "(unset: the slot takes the phone's capability as a bare bearer)"
	}
	raw, _ := env["TRAINER_INVOKER_SERVICE_ACCOUNT"].(string)
	got := invokersOf(raw)
	checks := []check{
		{Name: "image.env.TEE_REQUEST_SEAL", OK: seal == cfg.requestSeal, Detail: sealDetail},
		{Name: "image.env.TRAINER_INVOKER_SERVICE_ACCOUNT", OK: slices.Equal(got, cfg.trainerInvokers),
			Detail: fmt.Sprintf("%s (expected exactly %s)", strings.Join(got, ", "), strings.Join(cfg.trainerInvokers, ", "))},
	}
	for _, name := range reportedEnv {
		value, _ := env[name].(string)
		if value == "" {
			value = "(unset: the image's default)"
		}
		checks = append(checks, check{Name: "image.env." + name, OK: true, Detail: value + " (reported, not checked)"})
	}
	return checks
}

// invokersOf reads TRAINER_INVOKER_SERVICE_ACCOUNT as the enclave does
// (tee_mode.py TeeConfig): comma-separated, trimmed, lowercased; sorted
// and without repeats here.
func invokersOf(value string) []string {
	var out []string
	for _, account := range splitTrim(strings.ToLower(value)) {
		if !slices.Contains(out, account) {
			out = append(out, account)
		}
	}
	slices.Sort(out)
	return out
}

// provenanceCheck runs slsa-verifier over the running digest in the public
// registry: the provenance must name -source-uri at the token's release
// tag, and its commit must be the one the image stamp names. An image with
// no stamp, or a provenance whose commit cannot be read, fails.
func provenanceCheck(ctx context.Context, cfg *config, digest string, rel *release) (bool, string) {
	if rel == nil || !validRelease(rel.Version) || rel.Commit == "" {
		return false, "the image carries no release stamp (TEE_IMAGE_VERSION, TEE_IMAGE_COMMIT) to hold the provenance to"
	}
	args := []string{"verify-image", cfg.imageRepo + "@" + digest, "--source-uri", cfg.sourceURI,
		"--source-tag", rel.Version, "--print-provenance"}
	cmd := exec.CommandContext(ctx, cfg.slsaVerifierBin, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		if errors.Is(err, exec.ErrNotFound) || errors.Is(err, fs.ErrNotExist) {
			return false, "not checked: " + cfg.slsaVerifierBin + " not found (install slsa-verifier, VERIFY.md, or pass -slsa-verifier)"
		}
		said := strings.TrimSpace(stderr.String() + " " + stdout.String())
		if said == "" {
			said = err.Error()
		}
		return false, truncate(said, 300)
	}
	return provenanceVerdict(cfg.sourceURI, stdout.Bytes(), rel)
}

// provenanceVerdict holds what slsa-verifier printed to the image stamp.
func provenanceVerdict(sourceURI string, printed []byte, rel *release) (bool, string) {
	commit, ref := provenanceSource(printed)
	detail := sourceURI
	if ref != "" {
		detail += "@" + ref
	}
	if commit == "" {
		return false, detail + " (the provenance names no source commit)"
	}
	detail += " commit " + commit
	if commit != rel.Commit {
		return false, detail + " (the image stamp says " + rel.Commit + ")"
	}
	return true, detail
}

// provenanceSource pulls the source commit and ref out of the provenance
// slsa-verifier prints: SLSA v1 (buildDefinition.resolvedDependencies with
// a gitCommit digest, externalParameters.workflow.ref) or v0.2
// (invocation.configSource, materials).
func provenanceSource(out []byte) (commit, ref string) {
	i := bytes.IndexByte(out, '{')
	if i < 0 {
		return "", ""
	}
	var doc map[string]any
	if err := json.Unmarshal(bytes.TrimSpace(out[i:]), &doc); err != nil {
		return "", ""
	}
	pred, _ := doc["predicate"].(map[string]any)
	refOf := func(uri string) string {
		if j := strings.Index(uri, "@refs/tags/"); j >= 0 {
			return uri[j+len("@refs/tags/"):]
		}
		if j := strings.Index(uri, "@refs/heads/"); j >= 0 {
			return uri[j+len("@refs/heads/"):]
		}
		return ""
	}
	if bd, ok := pred["buildDefinition"].(map[string]any); ok {
		deps, _ := bd["resolvedDependencies"].([]any)
		for _, d := range deps {
			m, _ := d.(map[string]any)
			dg, _ := m["digest"].(map[string]any)
			if c, _ := dg["gitCommit"].(string); c != "" {
				commit = c
				uri, _ := m["uri"].(string)
				ref = refOf(uri)
				break
			}
		}
		if ref == "" {
			if wf, ok := nestedAny(bd, "externalParameters", "workflow").(map[string]any); ok {
				if r, _ := wf["ref"].(string); r != "" {
					ref = strings.TrimPrefix(strings.TrimPrefix(r, "refs/tags/"), "refs/heads/")
				}
			}
		}
		return commit, ref
	}
	if cs, ok := nestedAny(pred, "invocation", "configSource").(map[string]any); ok {
		uri, _ := cs["uri"].(string)
		ref = refOf(uri)
		if dg, ok := cs["digest"].(map[string]any); ok {
			commit, _ = dg["sha1"].(string)
		}
	}
	if commit == "" {
		mats, _ := pred["materials"].([]any)
		for _, mat := range mats {
			m, _ := mat.(map[string]any)
			dg, _ := m["digest"].(map[string]any)
			if c, _ := dg["sha1"].(string); c != "" {
				commit = c
				break
			}
		}
	}
	return commit, ref
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

// eatNonces is eat_nonce as a list: the token carries a single nonce as a
// string and several as an array.
func eatNonces(eatNonce any) []string {
	switch v := eatNonce.(type) {
	case string:
		return []string{v}
	case []any:
		out := make([]string, 0, len(v))
		for _, e := range v {
			s, _ := e.(string)
			out = append(out, s)
		}
		return out
	}
	return nil
}

func nonceMatches(eatNonce any, expected string) bool {
	return slices.Contains(eatNonces(eatNonce), expected)
}

// sameNonces reports whether got and want hold the same nonces, each as
// often, in any order.
func sameNonces(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	g, w := slices.Clone(got), slices.Clone(want)
	slices.Sort(g)
	slices.Sort(w)
	return slices.Equal(g, w)
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
