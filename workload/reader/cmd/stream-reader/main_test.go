package main

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/FemLed/masseuse-video-tee/workload/reader/record"
	"github.com/FemLed/masseuse-video-tee/workload/reader/testmedia"
)

// recordSink collects records as run writes them.
type recordSink struct {
	mu      sync.Mutex
	buf     []byte
	length  int
	headers []record.Header
	got     chan struct{}
}

func (s *recordSink) Write(b []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.buf = append(s.buf, b...)
	for len(s.buf) >= s.length {
		h, err := record.Unmarshal(s.buf[:record.Size])
		if err != nil {
			return 0, err
		}
		s.headers = append(s.headers, h)
		s.buf = s.buf[s.length:]
		select {
		case s.got <- struct{}{}:
		default:
		}
	}
	return len(b), nil
}

func (s *recordSink) snapshot() []record.Header {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]record.Header(nil), s.headers...)
}

func TestRecordsCarryTheSendersTimeEndToEnd(t *testing.T) {
	ffmpeg := testmedia.FFmpeg(t)
	units := testmedia.H264Units(t, 30, 10)
	s := testmedia.StartSender(t, units, nil)
	sink := &recordSink{length: record.Size + record.FrameLength(160, 120), got: make(chan struct{}, 1)}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		done <- run(ctx, config{url: s.URL(), width: 160, height: 120, ffmpeg: ffmpeg, ffmpegLog: "error",
			timeout: 10 * time.Second, out: sink, logf: t.Logf, dial: s.Net.Dial})
	}()
	deadline := time.After(20 * time.Second)
	for len(sink.snapshot()) < 25 {
		select {
		case <-sink.got:
		case <-deadline:
			t.Fatalf("%d records within 20 s", len(sink.snapshot()))
		}
	}
	cancel()
	if err := <-done; err != nil {
		t.Fatalf("run: %v", err)
	}
	headers := sink.snapshot()
	if headers[0].Flags&record.FlagKeyframe == 0 {
		t.Fatal("the first record is not a keyframe")
	}
	timed := 0
	for i, h := range headers {
		if h.Seq != uint32(i) || h.Width != 160 || h.Height != 120 || h.Codec != record.CodecH264 {
			t.Fatalf("record %d: %+v", i, h)
		}
		if i > 0 && h.RTPTs-headers[i-1].RTPTs != 3000 {
			t.Fatalf("record %d: RTP step %d", i, h.RTPTs-headers[i-1].RTPTs)
		}
		if h.Flags&record.FlagNTPValid == 0 {
			// The first units may go out before the sender's report reached
			// the reader (the hold gives up after a second on a busy
			// machine); every unit after it carries the time.
			if timed > 0 {
				t.Fatalf("record %d has no sender time after %d timed ones", i, timed)
			}
			continue
		}
		timed++
		want := s.UnitTime(h.RTPTs)
		if d := time.Unix(0, h.NTPNs).Sub(want); d > time.Millisecond || d < -time.Millisecond {
			t.Fatalf("record %d timed %s, sender said %s (off by %s)", i, time.Unix(0, h.NTPNs), want, d)
		}
	}
	if timed < len(headers)/2 {
		t.Fatalf("only %d of %d records carry the sender's time", timed, len(headers))
	}
}
