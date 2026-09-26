package decode

import (
	"bytes"
	"context"
	"testing"
	"time"

	"github.com/FemLed/masseuse-video-tee/workload/reader/record"
	"github.com/FemLed/masseuse-video-tee/workload/reader/source"
	"github.com/FemLed/masseuse-video-tee/workload/reader/testmedia"
)

// records splits the decoder's output into headers and frames.
func records(t *testing.T, out []byte, width, height int) []record.Header {
	t.Helper()
	length := record.FrameLength(width, height)
	var headers []record.Header
	for len(out) > 0 {
		if len(out) < record.Size+length {
			t.Fatalf("%d trailing bytes", len(out))
		}
		h, err := record.Unmarshal(out[:record.Size])
		if err != nil {
			t.Fatal(err)
		}
		frame := out[record.Size : record.Size+length]
		// The test pattern is not black: a decoded frame has light in it.
		var sum int
		for _, b := range frame[:width*height] {
			sum += int(b)
		}
		if sum/(width*height) < 16 {
			t.Fatalf("frame %d is dark: mean luma %d", h.Seq, sum/(width*height))
		}
		headers = append(headers, h)
		out = out[record.Size+length:]
	}
	return headers
}

func TestEveryUnitComesBackAsItsOwnFrame(t *testing.T) {
	ffmpeg := testmedia.FFmpeg(t)
	units := testmedia.H264Units(t, 45, 15)
	var out bytes.Buffer
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	dec, err := New(ctx, &out, Options{FFmpeg: ffmpeg, Width: 160, Height: 120, Codec: source.H264,
		Logf: t.Logf})
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	base := time.Date(2026, 9, 10, 12, 0, 0, 0, time.UTC)
	for i, nalus := range units {
		au := &source.AccessUnit{
			Seq: uint32(i), NALUs: nalus, PTS: int64(i) * 3000, RTPTs: 1_000_000 + uint32(i)*3000,
			Keyframe: i%15 == 0,
		}
		if i >= 3 { // the sender's report arrives after the third unit
			au.NTP, au.NTPValid = base.Add(time.Duration(i)*time.Second/30), true
		}
		if err := dec.Push(au); err != nil {
			t.Fatalf("push %d: %v", i, err)
		}
	}
	dec.Flush(10 * time.Second)
	if err := dec.Err(); err != nil {
		t.Fatal(err)
	}
	headers := records(t, out.Bytes(), 160, 120)
	if len(headers) != len(units) {
		t.Fatalf("%d records for %d units", len(headers), len(units))
	}
	for i, h := range headers {
		if h.Seq != uint32(i) || h.RTPTs != 1_000_000+uint32(i)*3000 {
			t.Fatalf("record %d is unit %d (rtp %d)", i, h.Seq, h.RTPTs)
		}
		if h.Width != 160 || h.Height != 120 || h.Codec != record.CodecH264 {
			t.Fatalf("record %d: %+v", i, h)
		}
		if key := h.Flags&record.FlagKeyframe != 0; key != (i%15 == 0) {
			t.Fatalf("record %d keyframe %v", i, key)
		}
		if valid := h.Flags&record.FlagNTPValid != 0; valid != (i >= 3) {
			t.Fatalf("record %d ntp valid %v", i, valid)
		}
		if i >= 3 && h.NTPNs != base.Add(time.Duration(i)*time.Second/30).UnixNano() {
			t.Fatalf("record %d ntp %d", i, h.NTPNs)
		}
	}
	st := dec.Stats()
	if st.Units != 45 || st.Frames != 45 || st.Unpaired != 0 || st.Dropped != 0 {
		t.Fatalf("stats %+v", st)
	}
}

// PipelineDelay is how many units ffmpeg holds between a unit going in and
// its frame coming out, by construction: an MPEG-TS video packet's end is
// known only when the next one starts (one), the H.264 parser closes a unit
// on the next unit's start (one more), and the encoder-muxer stage keeps
// one. The producer's overlay pays it as latency, not as timing: the
// record's time is the unit's whatever the delay.
const PipelineDelay = 3

func TestFramesArriveAsUnitsArePushed(t *testing.T) {
	// Live use: units come one at a time, a frame apart, and frames must
	// keep up rather than gather in a buffer. With the last unit pushed
	// and nothing flushed, every frame but the last PipelineDelay is out;
	// the flush brings those.
	ffmpeg := testmedia.FFmpeg(t)
	units := testmedia.H264Units(t, 30, 10)
	r, w := newPipe()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	dec, err := New(ctx, w, Options{FFmpeg: ffmpeg, Width: 160, Height: 120, Codec: source.H264, Logf: t.Logf})
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	length := record.Size + record.FrameLength(160, 120)
	recs := r.records(length)
	for i, nalus := range units {
		au := &source.AccessUnit{Seq: uint32(i), NALUs: nalus, PTS: int64(i) * 3000, Keyframe: i%10 == 0}
		if err := dec.Push(au); err != nil {
			t.Fatalf("push %d: %v", i, err)
		}
		time.Sleep(33 * time.Millisecond)
	}
	var got []uint32
	deadline := time.After(2 * time.Second)
	for len(got) < len(units)-PipelineDelay {
		select {
		case rec := <-recs:
			h, err := record.Unmarshal(rec)
			if err != nil {
				t.Fatal(err)
			}
			got = append(got, h.Seq)
		case <-deadline:
			t.Fatalf("frames out with the last unit pushed: %v", got)
		}
	}
	for i, seq := range got {
		if seq != uint32(i) {
			t.Fatalf("frames out of order: %v", got)
		}
	}
	// The rest come with the end of the input.
	dec.Flush(5 * time.Second)
	deadline = time.After(2 * time.Second)
	for len(got) < len(units) {
		select {
		case rec := <-recs:
			h, _ := record.Unmarshal(rec)
			if h.Seq != uint32(len(got)) {
				t.Fatalf("after %d frames came unit %d", len(got), h.Seq)
			}
			got = append(got, h.Seq)
		case <-deadline:
			t.Fatalf("%d frames after the flush", len(got))
		}
	}
}

func TestParseFrameLine(t *testing.T) {
	fl, ok := parseFrameLine("frame:12   pts:129000  pts_time:1.433333")
	if !ok || fl.index != 12 || fl.pts != 129000 {
		t.Fatalf("%+v %v", fl, ok)
	}
	if _, ok := parseFrameLine("frame:3    pts:NOPTS   pts_time:NOPTS"); ok {
		t.Fatal("NOPTS parsed")
	}
	if _, ok := parseFrameLine("r=1"); ok {
		t.Fatal("key line parsed")
	}
}

func TestPairingDropsWhatTheDecoderSkipped(t *testing.T) {
	d := &Decoder{}
	for i := 0; i < 5; i++ {
		d.pending = append(d.pending, &pendingUnit{au: &source.AccessUnit{Seq: uint32(i)}, pts: int64(i) * 3000})
	}
	// The decoder gave no frame for units 0 and 1 (before its first
	// keyframe, say): pairing 2 drops them.
	if au := d.pair(6000); au == nil || au.Seq != 2 {
		t.Fatalf("paired %+v", au)
	}
	if d.stats.Dropped != 2 || len(d.pending) != 2 {
		t.Fatalf("dropped %d, pending %d", d.stats.Dropped, len(d.pending))
	}
	// An unknown timestamp takes the oldest pending unit and is counted.
	if au := d.pair(99); au == nil || au.Seq != 3 || d.stats.Unpaired != 1 {
		t.Fatalf("paired %+v, unpaired %d", au, d.stats.Unpaired)
	}
	if au := d.pair(12000); au == nil || au.Seq != 4 {
		t.Fatalf("paired %+v", au)
	}
	if au := d.pair(15000); au != nil {
		t.Fatalf("paired %+v with nothing pending", au)
	}
}
