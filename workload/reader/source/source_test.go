package source

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/bluenviron/gortsplib/v5"
	"github.com/bluenviron/mediacommon/v2/pkg/codecs/h264"

	"github.com/Masseuse-ai/masseuse-video-tee/workload/reader/testmedia"
)

// collect runs a source until `want` units arrived or the timeout.
func collect(t *testing.T, s *testmedia.Sender, opts Options, want int) []*AccessUnit {
	t.Helper()
	got, _ := collectUntil(t, s, opts, func(got []*AccessUnit) bool { return len(got) >= want })
	return got
}

// collectUntil runs a source until `enough` says the units so far are, or
// the timeout, and returns them with the source's stats.
func collectUntil(t *testing.T, s *testmedia.Sender, opts Options, enough func([]*AccessUnit) bool) ([]*AccessUnit, Stats) {
	t.Helper()
	opts.Dial = s.Net.Dial
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	src, err := Open(ctx, s.URL(), opts)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer src.Close()
	if src.Codec() != H264 {
		t.Fatalf("codec %s", src.Codec())
	}
	var got []*AccessUnit
	done := errors.New("enough")
	err = src.Run(ctx, func(au *AccessUnit) error {
		got = append(got, au)
		if enough(got) {
			return done
		}
		return nil
	})
	if !errors.Is(err, done) {
		t.Fatalf("run: %v after %d units", err, len(got))
	}
	return got, src.Stats()
}

// through is an `enough` for collectUntil: the units reached the sender's
// unit k.
func through(k int) func([]*AccessUnit) bool {
	return func(got []*AccessUnit) bool { return got[len(got)-1].RTPTs >= testmedia.UnitRTPTs(k) }
}

func TestUnitsCarryTheSendersTime(t *testing.T) {
	units := testmedia.H264Units(t, 30, 10)
	s := testmedia.StartSender(t, units, nil)
	if w, h := func() (int, int) {
		src, err := Open(context.Background(), s.URL(), Options{Dial: s.Net.Dial})
		if err != nil {
			t.Fatal(err)
		}
		defer src.Close()
		return src.Size()
	}(); w != testmedia.Width || h != testmedia.Height {
		t.Fatalf("size %dx%d", w, h)
	}
	// The sender's report can take a moment to reach the client on a busy
	// machine: the hold is long enough here that the first unit still
	// gets its time (a report that never comes is the other test).
	got, st := collectUntil(t, s, Options{Hold: 10 * time.Second}, func(got []*AccessUnit) bool { return len(got) >= 40 })
	if !got[0].Keyframe {
		t.Fatal("the first unit is not a keyframe")
	}
	if st != (Stats{}) {
		t.Fatalf("stats of a clean stream: %+v", st)
	}
	for i, au := range got {
		if au.Seq != uint32(i) {
			t.Fatalf("unit %d has seq %d", i, au.Seq)
		}
		if au.Damaged {
			t.Fatalf("unit %d of a clean stream is damaged", i)
		}
		if i > 0 && au.PTS-got[i-1].PTS != 3000 {
			t.Fatalf("unit %d: PTS step %d", i, au.PTS-got[i-1].PTS)
		}
		if i > 0 && au.RTPTs-got[i-1].RTPTs != 3000 {
			t.Fatalf("unit %d: RTP step %d", i, au.RTPTs-got[i-1].RTPTs)
		}
		if !au.NTPValid {
			t.Fatalf("unit %d has no sender time", i)
		}
		if d := au.NTP.Sub(s.UnitTime(au.RTPTs)); d > time.Millisecond || d < -time.Millisecond {
			t.Fatalf("unit %d timed %s, sender said %s (off by %s)", i, au.NTP, s.UnitTime(au.RTPTs), d)
		}
		if au.Keyframe && h264.NALUType(au.NALUs[0][0]&0x1F) != h264.NALUTypeSPS {
			t.Fatalf("keyframe %d does not start with its parameter sets", i)
		}
	}
	keyframes := 0
	for _, au := range got {
		if au.Keyframe {
			keyframes++
		}
	}
	if keyframes < 3 {
		t.Fatalf("%d keyframes in %d units", keyframes, len(got))
	}
}

func TestParameterSetsFromTheSDPPrecedeKeyframes(t *testing.T) {
	units := testmedia.H264Units(t, 20, 10)
	// The sender's units carry no parameter sets of their own; the SDP's
	// are put before each keyframe so a decoder can start there.
	var bare [][][]byte
	for _, u := range units {
		var nalus [][]byte
		for _, n := range u {
			switch h264.NALUType(n[0] & 0x1F) {
			case h264.NALUTypeSPS, h264.NALUTypePPS:
				continue
			}
			nalus = append(nalus, n)
		}
		bare = append(bare, nalus)
	}
	// The SDP still carries them, from the full units.
	s := testmedia.StartSender(t, bare, func(s *testmedia.Sender, _ *gortsplib.Server) { s.SDPFrom = units[0] })
	got := collect(t, s, Options{}, 12)
	for i, au := range got {
		types := []h264.NALUType{}
		for _, n := range au.NALUs {
			types = append(types, h264.NALUType(n[0]&0x1F))
		}
		if au.Keyframe {
			if len(types) < 3 || types[0] != h264.NALUTypeSPS || types[1] != h264.NALUTypePPS {
				t.Fatalf("keyframe %d: %v", i, types)
			}
		} else if types[0] == h264.NALUTypeSPS {
			t.Fatalf("unit %d is not a keyframe but got parameter sets: %v", i, types)
		}
	}
}

func TestASenderThatDoesNotReportFlowsArrivalTimed(t *testing.T) {
	units := testmedia.H264Units(t, 20, 10)
	s := testmedia.StartSender(t, units, func(_ *testmedia.Sender, srv *gortsplib.Server) { srv.DisableRTCPSenderReports = true })
	started := time.Now()
	got := collect(t, s, Options{Hold: 300 * time.Millisecond}, 15)
	if time.Since(started) < 300*time.Millisecond {
		t.Fatal("units came before the hold ran out")
	}
	for i, au := range got {
		if au.NTPValid || !au.NTP.IsZero() {
			t.Fatalf("unit %d claims a sender time without a report", i)
		}
		if au.Seq != uint32(i) {
			t.Fatalf("unit %d has seq %d", i, au.Seq)
		}
	}
}

func TestALostPacketDamagesItsUnit(t *testing.T) {
	units := testmedia.H264Units(t, 30, 10)
	var dropped atomic.Int32
	s := testmedia.StartSender(t, units, func(s *testmedia.Sender, _ *gortsplib.Server) {
		s.WaitForPlay = true
		// One packet from the middle of the keyframe played 20th: the unit
		// cannot be had whole.
		s.Drop = func(unit, packet, packets int) bool {
			if unit == 20 && packets >= 3 && packet == packets/2 {
				dropped.Add(1)
				return true
			}
			return false
		}
	})
	got, st := collectUntil(t, s, Options{Hold: 10 * time.Second}, through(24))
	if dropped.Load() != 1 {
		t.Fatalf("the sender dropped %d packets of the keyframe", dropped.Load())
	}
	if st.PacketsLost != 1 || st.Damaged != 1 {
		t.Fatalf("stats %+v", st)
	}
	// One unit is damaged: the keyframe, or - when the lost packet cost the
	// unit its end, so that it never came out - the unit after it.
	for i, au := range got {
		if !au.Damaged {
			continue
		}
		if au.RTPTs != testmedia.UnitRTPTs(20) && au.RTPTs != testmedia.UnitRTPTs(21) {
			t.Fatalf("unit %d at RTP %d is damaged", i, au.RTPTs)
		}
		if i > 0 && got[i-1].RTPTs >= testmedia.UnitRTPTs(20) {
			t.Fatalf("unit %d at RTP %d came out whole after the loss", i-1, got[i-1].RTPTs)
		}
	}
}

func TestALostUnitDamagesTheOneAfterIt(t *testing.T) {
	units := testmedia.H264Units(t, 30, 10)
	var dropped atomic.Int32
	s := testmedia.StartSender(t, units, func(s *testmedia.Sender, _ *gortsplib.Server) {
		s.WaitForPlay = true
		// Every packet of the unit played 13th, between keyframes: it never
		// comes out, and the unit after it refers to a picture that is gone.
		s.Drop = func(unit, packet, packets int) bool {
			if unit == 13 {
				dropped.Add(1)
				return true
			}
			return false
		}
	})
	got, st := collectUntil(t, s, Options{Hold: 10 * time.Second}, through(16))
	if dropped.Load() == 0 || st.PacketsLost != uint64(dropped.Load()) || st.Damaged != 1 {
		t.Fatalf("the sender dropped %d packets; stats %+v", dropped.Load(), st)
	}
	saw := false
	for i, au := range got {
		if au.RTPTs == testmedia.UnitRTPTs(13) {
			t.Fatalf("unit %d is the lost one", i)
		}
		next := au.RTPTs == testmedia.UnitRTPTs(14)
		saw = saw || next
		if au.Damaged != next {
			t.Fatalf("unit %d at RTP %d: damaged %v", i, au.RTPTs, au.Damaged)
		}
	}
	if !saw {
		t.Fatal("the unit after the lost one did not come out")
	}
}

func TestNoVideoTrackIsAnError(t *testing.T) {
	s := testmedia.StartSender(t, nil, func(s *testmedia.Sender, _ *gortsplib.Server) { s.AudioOnly = true })
	_, err := Open(context.Background(), s.URL(), Options{Dial: s.Net.Dial})
	if !errors.Is(err, ErrNoVideo) {
		t.Fatalf("open: %v", err)
	}
}
