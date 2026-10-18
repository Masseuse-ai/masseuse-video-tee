package main

// -policy-url: the values a service holds its slots to, published as a JSON
// document (VERIFY.md, "Which values to verify against"), in place of the
// defaults, which are the operator's own fleet. The document is either those
// values or an object that carries them as its "verify" member. A value it
// leaves out, and every flag given on the command line, keeps the flag's
// value. What it cannot change: the token's issuer and keys, the hardware,
// the production posture, and that a signature and the provenance are
// required.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

const maxPolicyBytes = 1 << 20

type slotPolicy struct {
	SourceURI        *string  `json:"sourceUri"`
	ImageRepo        *string  `json:"imageRepo"`
	SignerKeyIDs     []string `json:"signerKeyIds"`
	MinRelease       *string  `json:"minRelease"`
	ReadingsURL      *string  `json:"readingsUrl"`
	Invokers         []string `json:"invokers"`
	CaptureBucket    *string  `json:"captureBucket"`
	RequestSeal      *string  `json:"requestSeal"`
	ProjectIDs       []string `json:"projectIds"`
	ImageRefPrefixes []string `json:"imageRefPrefixes"`
}

// readPolicy fetches the document at an https:// URL, or reads it from a
// file path (file:// or bare).
func readPolicy(ctx context.Context, client *http.Client, location string) (*slotPolicy, error) {
	var body []byte
	switch {
	case strings.HasPrefix(location, "https://"):
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, location, nil)
		if err != nil {
			return nil, err
		}
		req.Header.Set("Accept", "application/json")
		resp, err := client.Do(req)
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("%s answered %s", location, resp.Status)
		}
		if body, err = io.ReadAll(io.LimitReader(resp.Body, maxPolicyBytes+1)); err != nil {
			return nil, err
		}
	case strings.HasPrefix(location, "http://"):
		return nil, errors.New("-policy-url must be https:// or a file")
	default:
		var err error
		if body, err = os.ReadFile(strings.TrimPrefix(location, "file://")); err != nil {
			return nil, err
		}
	}
	if len(body) > maxPolicyBytes {
		return nil, fmt.Errorf("the policy is larger than %d bytes", maxPolicyBytes)
	}
	return parsePolicy(body)
}

func parsePolicy(body []byte) (*slotPolicy, error) {
	var outer map[string]json.RawMessage
	if err := json.Unmarshal(body, &outer); err != nil {
		return nil, fmt.Errorf("the policy is not a JSON object: %w", err)
	}
	values := json.RawMessage(body)
	if v, ok := outer["verify"]; ok {
		if string(bytes.TrimSpace(v)) == "null" {
			return nil, errors.New("the policy names no values to verify against (its verify is null)")
		}
		values = v
	}
	var p slotPolicy
	if err := json.Unmarshal(values, &p); err != nil {
		return nil, fmt.Errorf("the policy's values: %w", err)
	}
	return &p, nil
}

// applyPolicy takes each value the policy names into cfg, unless its flag
// was given on the command line (set).
func applyPolicy(cfg *config, p *slotPolicy, set map[string]bool) {
	str := func(name string, from *string, to *string) {
		if from != nil && !set[name] {
			*to = strings.TrimSpace(*from)
		}
	}
	list := func(name string, from []string, to *[]string) {
		if from != nil && !set[name] {
			*to = splitTrim(strings.Join(from, ","))
		}
	}
	str("source-uri", p.SourceURI, &cfg.sourceURI)
	str("image-repo", p.ImageRepo, &cfg.imageRepo)
	str("min-release", p.MinRelease, &cfg.minRelease)
	str("trainer-url", p.ReadingsURL, &cfg.trainerURL)
	str("capture-bucket", p.CaptureBucket, &cfg.captureBucket)
	str("request-seal", p.RequestSeal, &cfg.requestSeal)
	list("signer-key-ids", p.SignerKeyIDs, &cfg.signerKeyIDs)
	for i, k := range cfg.signerKeyIDs {
		cfg.signerKeyIDs[i] = strings.ToLower(k)
	}
	list("project-id", p.ProjectIDs, &cfg.projectIDs)
	list("image-ref-prefix", p.ImageRefPrefixes, &cfg.imageRefPrefixes)
	if p.Invokers != nil && !set["trainer-invoker"] {
		cfg.trainerInvokers = invokersOf(strings.Join(p.Invokers, ","))
	}
}
