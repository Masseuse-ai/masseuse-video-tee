package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/golang-jwt/jwt/v5"
)

// A service's published policy: its values as the "verify" member of a
// larger document, the way an API that leases slots serves them.
const servicePolicy = `{
  "object": "attestation_policy",
  "mode": "live",
  "verify": {
    "sourceUri": "github.com/Masseuse-ai/masseuse-video-tee",
    "imageRepo": "ghcr.io/masseuse-ai/masseuse-video-tee",
    "signerKeyIds": ["CAFE"],
    "minRelease": "v0.4.0",
    "readingsUrl": "https://readings.example",
    "invokers": ["Controller@Example.iam.gserviceaccount.com"],
    "captureBucket": "",
    "requestSeal": "v1",
    "projectIds": ["own-project"],
    "imageRefPrefixes": ["us-docker.pkg.dev/own-project/slots/"]
  }
}`

func TestReadsAServicePolicyOverHTTPS(t *testing.T) {
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("mode") != "live" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(servicePolicy))
	}))
	defer server.Close()
	p, err := readPolicy(context.Background(), server.Client(), server.URL+"/v1/attestation-policy?mode=live")
	if err != nil {
		t.Fatal(err)
	}
	if p.ReadingsURL == nil || *p.ReadingsURL != "https://readings.example" || !slices.Equal(p.ProjectIDs, []string{"own-project"}) {
		t.Fatalf("policy: %+v", p)
	}
	if _, err := readPolicy(context.Background(), server.Client(), strings.Replace(server.URL, "https://", "http://", 1)); err == nil {
		t.Fatal("a plain http:// policy must be refused")
	}
}

func TestReadsABareDocumentFromAFileAndRefusesANullOne(t *testing.T) {
	dir := t.TempDir()
	bare := filepath.Join(dir, "values.json")
	if err := os.WriteFile(bare, []byte(`{"captureBucket": "", "projectIds": ["a", "b"]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	p, err := readPolicy(context.Background(), http.DefaultClient, bare)
	if err != nil || p.CaptureBucket == nil || *p.CaptureBucket != "" || len(p.ProjectIDs) != 2 {
		t.Fatalf("bare document: %+v %v", p, err)
	}
	test := filepath.Join(dir, "test.json")
	if err := os.WriteFile(test, []byte(`{"mode": "test", "verify": null}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := readPolicy(context.Background(), http.DefaultClient, "file://"+test); err == nil || !strings.Contains(err.Error(), "null") {
		t.Fatalf("a null verify must be refused: %v", err)
	}
}

func TestThePolicyReplacesDefaultsAndAGivenFlagWins(t *testing.T) {
	p, err := parsePolicy([]byte(servicePolicy))
	if err != nil {
		t.Fatal(err)
	}
	cfg := &config{captureBucket: "masseuse-ai-prod", trainerURL: "https://trainer.example", projectIDs: []string{"prod-masseuse-video-tee"}}
	applyPolicy(cfg, p, map[string]bool{"trainer-url": true})
	if cfg.captureBucket != "" {
		t.Fatalf("the policy's empty record bucket must replace the default: %q", cfg.captureBucket)
	}
	if cfg.trainerURL != "https://trainer.example" {
		t.Fatalf("a flag given on the command line must win: %q", cfg.trainerURL)
	}
	if !slices.Equal(cfg.signerKeyIDs, []string{"cafe"}) || !slices.Equal(cfg.trainerInvokers, []string{"controller@example.iam.gserviceaccount.com"}) {
		t.Fatalf("keys and invokers are compared lower-cased: %v %v", cfg.signerKeyIDs, cfg.trainerInvokers)
	}
	if !slices.Equal(cfg.projectIDs, []string{"own-project"}) || cfg.minRelease != "v0.4.0" {
		t.Fatalf("projects and release: %v %q", cfg.projectIDs, cfg.minRelease)
	}
}

func TestPassesAgainstASlotInAnotherProjectWithItsValues(t *testing.T) {
	fs := newFakeSlot(t, func(c jwt.MapClaims) {
		sub := c["submods"].(map[string]any)
		sub["gce"] = map[string]any{"project_id": "own-project", "instance_id": "7"}
		container := sub["container"].(map[string]any)
		container["image_reference"] = "us-docker.pkg.dev/own-project/slots/masseuse-video-tee@" + testDigest
	})
	cfg := fs.config()
	if r := runReport(t, cfg); r.Passed {
		t.Fatal("the operator fleet's defaults must not pass a slot in another project")
	}
	cfg.projectIDs = []string{"prod-masseuse-video-tee", "own-project"}
	cfg.imageRefPrefixes = []string{"us-central1-docker.pkg.dev/prod-masseuse-video-tee/masseuse-video-tee/", "us-docker.pkg.dev/own-project/slots/"}
	if r := runReport(t, cfg); !r.Passed {
		t.Fatalf("expected pass with the project and registry named, failed: %v", failed(r))
	}
}
