// stream-reader reads one RTSP video track and writes frame records
// (workload/reader/record) to stdout: each frame decoded to yuv420p at a
// fixed size by an ffmpeg child, prefixed with the time the sender's RTCP
// sender reports give it. producer/producer.py runs one per view and pairs
// the views on that time.
//
//	stream-reader -url rtsp://127.0.0.1:8554/cam -width 1280 -height 720
//
// Between the stream and ffmpeg stands a gate (workload/reader/gate):
// reading never waits on the decoder, and a unit that arrived damaged, or
// that comes while the decoder is more than -queue behind, is left out with
// every unit after it up to the next whole keyframe, so that no frame is
// decoded from a reference that is missing. Every record's header counts
// what was lost and left out so far. -threads N decodes N frames at once,
// each thread past the first holding the picture a frame longer; the
// default, one, puts every frame out as soon as it can be.
//
// It exits 0 when the stream ends, non-zero on a failure; the producer
// reconnects either way. Diagnostics go to stderr, one line each.
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/Masseuse-ai/masseuse-video-tee/workload/reader/decode"
	"github.com/Masseuse-ai/masseuse-video-tee/workload/reader/gate"
	"github.com/Masseuse-ai/masseuse-video-tee/workload/reader/record"
	"github.com/Masseuse-ai/masseuse-video-tee/workload/reader/source"
)

func main() {
	cfg := config{out: os.Stdout}
	flag.StringVar(&cfg.url, "url", "", "the RTSP URL to read (required)")
	flag.IntVar(&cfg.width, "width", 0, "frame width every record carries (required, even)")
	flag.IntVar(&cfg.height, "height", 0, "frame height every record carries (required, even)")
	flag.StringVar(&cfg.ffmpeg, "ffmpeg", "ffmpeg", "the ffmpeg executable")
	flag.DurationVar(&cfg.timeout, "timeout", source.DefaultReadTimeout, "how long the stream may stay silent")
	flag.StringVar(&cfg.ffmpegLog, "ffmpeg-loglevel", "error", "ffmpeg's -loglevel")
	flag.DurationVar(&cfg.queue, "queue", gate.DefaultMaxWait,
		"how far the decoder may fall behind the stream before units are left out up to the next keyframe")
	flag.IntVar(&cfg.threads, "threads", 1,
		"frames decoded at once; each thread past the first holds the picture a frame longer")
	flag.Parse()
	if cfg.url == "" || cfg.width <= 0 || cfg.height <= 0 || cfg.queue <= 0 || cfg.threads < 1 {
		flag.Usage()
		os.Exit(2)
	}
	cfg.logf = func(format string, args ...any) {
		fmt.Fprintf(os.Stderr, "stream-reader: "+format+"\n", args...)
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	if err := run(ctx, cfg); err != nil {
		cfg.logf("%v", err)
		os.Exit(1)
	}
}

// config is one run's settings.
type config struct {
	url           string
	width, height int
	ffmpeg        string
	ffmpegLog     string
	timeout       time.Duration
	queue         time.Duration // 0 means gate.DefaultMaxWait
	threads       int           // 0 means one
	out           io.Writer
	logf          func(string, ...any)
	// dial replaces the network dial (tests).
	dial func(ctx context.Context, network, address string) (net.Conn, error)
}

// run reads the stream into records on cfg.out until it ends.
func run(ctx context.Context, cfg config) error {
	if cfg.logf == nil {
		cfg.logf = func(string, ...any) {}
	}
	src, err := source.Open(ctx, cfg.url, source.Options{ReadTimeout: cfg.timeout, Logf: cfg.logf, Dial: cfg.dial})
	if err != nil {
		return err
	}
	defer src.Close()
	w, h := src.Size()
	cfg.logf("%s: %s%s; frames %dx%d", cfg.url, src.Codec(), sizeNote(w, h), cfg.width, cfg.height)

	g := gate.New(gate.Options{MaxWait: cfg.queue, Logf: cfg.logf})
	out := &lockedWriter{w: bufio.NewWriterSize(cfg.out, 1<<20)}
	dec, err := decode.New(ctx, out, decode.Options{
		FFmpeg: cfg.ffmpeg, Width: cfg.width, Height: cfg.height, Codec: src.Codec(), Logf: cfg.logf,
		LogLevel: cfg.ffmpegLog, Threads: cfg.threads,
		Annotate: func(h *record.Header) {
			ss, gs := src.Stats(), g.Stats()
			h.Lost, h.Errors = record.Count32(ss.PacketsLost), record.Count32(ss.Errors)
			h.Dropped, h.Queued = record.Count32(gs.Dropped), record.Count16(gs.Queued)
		},
	})
	if err != nil {
		return err
	}
	defer dec.Close()
	g.Start(dec.Push)

	// Records leave as soon as they are complete: a frame held in the
	// buffer is latency the overlay pays.
	stopFlush := make(chan struct{})
	flushed := make(chan struct{})
	go func() {
		defer close(flushed)
		t := time.NewTicker(10 * time.Millisecond)
		defer t.Stop()
		for {
			select {
			case <-t.C:
				_ = out.Flush()
			case <-stopFlush:
				_ = out.Flush()
				return
			}
		}
	}()

	runErr := src.Run(ctx, g.Offer)
	g.Close(2 * time.Second)
	dec.Flush(2 * time.Second)
	close(stopFlush)
	<-flushed
	st, ss, gs := dec.Stats(), src.Stats(), g.Stats()
	cfg.logf("%s: ended (%v); units %d, left out %d, frames %d, unpaired %d, skipped %d; packets lost %d, errors %d",
		cfg.url, runErr, gs.Offered, gs.Dropped, st.Frames, st.Unpaired, st.Skipped, ss.PacketsLost, ss.Errors)
	if derr := dec.Err(); derr != nil {
		return derr
	}
	// The stream ending, or a signal, is how a session ends; the producer
	// decides whether to come back.
	return nil
}

func sizeNote(w, h int) string {
	if w == 0 || h == 0 {
		return ""
	}
	return fmt.Sprintf(" %dx%d", w, h)
}

// lockedWriter is a bufio.Writer the decoder writes and the flush ticker
// flushes.
type lockedWriter struct {
	mu sync.Mutex
	w  *bufio.Writer
}

func (l *lockedWriter) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.w.Write(p)
}

func (l *lockedWriter) Flush() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.w.Flush()
}
