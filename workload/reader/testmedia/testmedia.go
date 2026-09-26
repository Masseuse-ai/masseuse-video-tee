// Package testmedia gives the tests coded video and an in-process network:
// H.264 access units of ffmpeg's test pattern, and a listener/dialer pair
// over net.Pipe so an RTSP server and its client meet without a socket.
package testmedia

import (
	"context"
	"net"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/bluenviron/mediacommon/v2/pkg/codecs/h264"
)

// FFmpeg is the ffmpeg on PATH, or "" when there is none (the tests that
// need it skip).
func FFmpeg(t testing.TB) string {
	t.Helper()
	path, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Skip("ffmpeg not on PATH")
	}
	out, _ := exec.Command(path, "-hide_banner", "-encoders").Output()
	if !strings.Contains(string(out), "libx264") {
		t.Skip("ffmpeg without libx264")
	}
	return path
}

// H264Units encodes `frames` frames of the test pattern at 30 fps,
// Width x Height, with a keyframe every `gop` frames and an access unit
// delimiter starting each unit, and returns the units' NAL units.
func H264Units(t testing.TB, frames, gop int) [][][]byte {
	t.Helper()
	ffmpeg := FFmpeg(t)
	cmd := exec.Command(ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
		"-f", "lavfi", "-i", "testsrc=size=320x240:rate=30",
		"-frames:v", itoa(frames), "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
		"-x264-params", "aud=1:keyint="+itoa(gop)+":min-keyint="+itoa(gop)+":scenecut=0:repeat-headers=1",
		"-f", "h264", "-")
	raw, err := cmd.Output()
	if err != nil {
		t.Fatalf("ffmpeg: %v", err)
	}
	nalus := splitAnnexB(raw)
	if len(nalus) == 0 {
		t.Fatal("annex b: no NAL units")
	}
	var units [][][]byte
	for _, n := range nalus {
		if h264.NALUType(n[0]&0x1F) == h264.NALUTypeAccessUnitDelimiter {
			units = append(units, nil)
			continue
		}
		if len(units) == 0 {
			units = append(units, nil)
		}
		units[len(units)-1] = append(units[len(units)-1], n)
	}
	if len(units) != frames {
		t.Fatalf("%d units for %d frames", len(units), frames)
	}
	return units
}

// Width and Height of H264Units's pictures.
const (
	Width  = 320
	Height = 240
)

func itoa(n int) string { return strconv.Itoa(n) }

// splitAnnexB cuts an Annex B byte stream at its start codes (three or
// four bytes) into NAL units without them.
func splitAnnexB(raw []byte) [][]byte {
	var starts []int
	for i := 0; i+3 <= len(raw); i++ {
		if raw[i] == 0 && raw[i+1] == 0 && raw[i+2] == 1 {
			starts = append(starts, i+3)
			i += 2
		}
	}
	var nalus [][]byte
	for k, start := range starts {
		end := len(raw)
		if k+1 < len(starts) {
			end = starts[k+1] - 3
			// a four-byte start code has a zero before the three
			if end > start && raw[end-1] == 0 {
				end--
			}
		}
		if end > start {
			nalus = append(nalus, raw[start:end])
		}
	}
	return nalus
}

// PipeNet is an in-process network: Listen returns its listener, Dial
// connects to it. The RTSP server reads TCP addresses off its connections,
// so both ends of a pipe carry loopback ones.
type PipeNet struct {
	conns  chan net.Conn
	closed chan struct{}
	once   sync.Once
	seq    int
	mu     sync.Mutex
}

// NewPipeNet returns a PipeNet.
func NewPipeNet() *PipeNet {
	return &PipeNet{conns: make(chan net.Conn, 8), closed: make(chan struct{})}
}

var listenAddr = &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 8554}

// Listen is for gortsplib.Server.Listen.
func (p *PipeNet) Listen(string, string) (net.Listener, error) { return p, nil }

// Dial is for gortsplib.Client.DialContext.
func (p *PipeNet) Dial(ctx context.Context, _, _ string) (net.Conn, error) {
	p.mu.Lock()
	p.seq++
	peer := &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 40000 + p.seq%20000}
	p.mu.Unlock()
	a, b := net.Pipe()
	select {
	case p.conns <- &addrConn{Conn: b, local: listenAddr, remote: peer}:
		return &addrConn{Conn: a, local: peer, remote: listenAddr}, nil
	case <-p.closed:
		a.Close()
		b.Close()
		return nil, net.ErrClosed
	case <-ctx.Done():
		a.Close()
		b.Close()
		return nil, ctx.Err()
	}
}

// addrConn gives a pipe end TCP addresses.
type addrConn struct {
	net.Conn
	local, remote net.Addr
}

func (c *addrConn) LocalAddr() net.Addr  { return c.local }
func (c *addrConn) RemoteAddr() net.Addr { return c.remote }

// Accept implements net.Listener.
func (p *PipeNet) Accept() (net.Conn, error) {
	select {
	case c := <-p.conns:
		return c, nil
	case <-p.closed:
		return nil, net.ErrClosed
	}
}

// Close implements net.Listener.
func (p *PipeNet) Close() error {
	p.once.Do(func() { close(p.closed) })
	return nil
}

// Addr implements net.Listener.
func (p *PipeNet) Addr() net.Addr { return listenAddr }
