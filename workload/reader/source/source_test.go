package source

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/bluenviron/gortsplib/v5"
	"github.com/bluenviron/mediacommon/v2/pkg/codecs/h264"

	"github.com/FemLed/masseuse-video-tee/workload/reader/testmedia"
)

// collect runs a source until `want` units arrived or the timeout.
func collect(t *testing.T, s *testmedia.Sender, opts Options, want int) []*AccessUnit {
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
	enough := errors.New("enough")
	err = src.Run(ctx, func(au *AccessUnit) error {
		got = append(got, au)
		if len(got) >= want {
			return enough
		}
		return nil
	})
	if !errors.Is(err, enough) {
		t.Fatalf("run: %v after %d units", err, len(got))
	}
	return got
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
	got := collect(t, s, Options{Hold: 10 * time.Second}, 40)
	if !got[0].Keyframe {
		t.Fatal("the first unit is not a keyframe")
	}
	for i, au := range got {
		if au.Seq != uint32(i) {
			t.Fatalf("unit %d has seq %d", i, au.Seq)
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

func TestNoVideoTrackIsAnError(t *testing.T) {
	s := testmedia.StartSender(t, nil, func(s *testmedia.Sender, _ *gortsplib.Server) { s.AudioOnly = true })
	_, err := Open(context.Background(), s.URL(), Options{Dial: s.Net.Dial})
	if !errors.Is(err, ErrNoVideo) {
		t.Fatalf("open: %v", err)
	}
}
