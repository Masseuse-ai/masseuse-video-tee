package main

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/bluenviron/gortsplib/v5"

	"github.com/Masseuse-ai/masseuse-video-tee/workload/reader/record"
	"github.com/Masseuse-ai/masseuse-video-tee/workload/reader/testmedia"
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
		if h.Lost != 0 || h.Errors != 0 || h.Dropped != 0 || h.Skipped != 0 {
			t.Fatalf("record %d of a clean stream counts losses: %+v", i, h)
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

func TestALostUnitTakesTheRestOfItsGroupWithIt(t *testing.T) {
	// Every packet of the unit played 13th is lost. Units 14 to 19 refer
	// to it, so no frame comes out for 13 to 19 - not a smeared one - and
	// the next record is the keyframe at 20.
	ffmpeg := testmedia.FFmpeg(t)
	units := testmedia.H264Units(t, 30, 10)
	s := testmedia.StartSender(t, units, func(s *testmedia.Sender, _ *gortsplib.Server) {
		s.WaitForPlay = true
		s.Drop = func(unit, _, _ int) bool { return unit == 13 }
	})
	sink := &recordSink{length: record.Size + record.FrameLength(160, 120), got: make(chan struct{}, 1)}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		done <- run(ctx, config{url: s.URL(), width: 160, height: 120, ffmpeg: ffmpeg, ffmpegLog: "error",
			timeout: 10 * time.Second, out: sink, logf: t.Logf, dial: s.Net.Dial})
	}()
	deadline := time.After(20 * time.Second)
	for {
		headers := sink.snapshot()
		if len(headers) > 0 && headers[len(headers)-1].RTPTs >= testmedia.UnitRTPTs(23) {
			break
		}
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
	resume := -1
	for i, h := range headers {
		if h.RTPTs >= testmedia.UnitRTPTs(13) && h.RTPTs < testmedia.UnitRTPTs(20) {
			t.Fatalf("record %d is the unit at RTP %d", i, h.RTPTs)
		}
		if resume < 0 && h.RTPTs >= testmedia.UnitRTPTs(20) {
			resume = i
		}
	}
	if resume < 1 {
		t.Fatalf("no record before or after the loss: %d", resume)
	}
	before, after := headers[resume-1], headers[resume]
	if before.RTPTs != testmedia.UnitRTPTs(12) || after.RTPTs != testmedia.UnitRTPTs(20) || after.Flags&record.FlagKeyframe == 0 {
		t.Fatalf("around the loss: %+v then %+v", before, after)
	}
	// The units left out are the gap in seq: 14 to 19 came, 13 never did.
	if after.Lost == 0 || after.Dropped != 6 || after.Seq-before.Seq-1 != after.Dropped {
		t.Fatalf("counts at the keyframe: %+v (before it %+v)", after, before)
	}
	for _, h := range headers[resume:] {
		if h.Lost != after.Lost || h.Dropped != after.Dropped || h.Skipped != 0 {
			t.Fatalf("counts moved after the keyframe: %+v", h)
		}
	}
}
