package decode

import (
	"bytes"
	"context"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/Masseuse-ai/masseuse-video-tee/workload/reader/record"
	"github.com/Masseuse-ai/masseuse-video-tee/workload/reader/source"
	"github.com/Masseuse-ai/masseuse-video-tee/workload/reader/testmedia"
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
	for _, threads := range []int{1, 2} {
		t.Run(fmt.Sprintf("threads=%d", threads), func(t *testing.T) { everyUnitComesBack(t, threads) })
	}
}

func everyUnitComesBack(t *testing.T, threads int) {
	ffmpeg := testmedia.FFmpeg(t)
	units := testmedia.H264Units(t, 45, 15)
	var out bytes.Buffer
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	// The caller's counts ride on every header beside the decoder's own.
	annotate := func(h *record.Header) { h.Lost, h.Dropped, h.Queued = 7, 5, 3 }
	dec, err := New(ctx, &out, Options{FFmpeg: ffmpeg, Width: 160, Height: 120, Codec: source.H264,
		Logf: t.Logf, Threads: threads, Annotate: annotate})
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
		if h.Lost != 7 || h.Dropped != 5 || h.Queued != 3 || h.Errors != 0 || h.Skipped != 0 {
			t.Fatalf("record %d counts %+v", i, h)
		}
		if int(h.Pending) > len(units)-i-1 {
			t.Fatalf("record %d: %d units pending with %d to come", i, h.Pending, len(units)-i-1)
		}
	}
	if last := headers[len(headers)-1]; last.Pending != 0 {
		t.Fatalf("the last record says %d units pending", last.Pending)
	}
	st := dec.Stats()
	if st.Units != 45 || st.Frames != 45 || st.Unpaired != 0 || st.Skipped != 0 {
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
	for _, threads := range []int{1, 2} {
		t.Run(fmt.Sprintf("threads=%d", threads), func(t *testing.T) { framesArriveAsPushed(t, threads) })
	}
}

func framesArriveAsPushed(t *testing.T, threads int) {
	// Live use: units come one at a time, a frame apart, and frames must
	// keep up rather than gather in a buffer. With the last unit pushed
	// and nothing flushed, every frame but the last PipelineDelay is out,
	// and one more for each decoding thread past the first; the flush
	// brings those.
	ffmpeg := testmedia.FFmpeg(t)
	units := testmedia.H264Units(t, 30, 10)
	r, w := newPipe()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	dec, err := New(ctx, w, Options{FFmpeg: ffmpeg, Width: 160, Height: 120, Codec: source.H264, Logf: t.Logf,
		Threads: threads})
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
	for len(got) < len(units)-PipelineDelay-(threads-1) {
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

func TestAPictureThatTurnsKeepsItsFramesComing(t *testing.T) {
	// The phone is turned: the sender's next keyframe has the other
	// orientation. ffmpeg rebuilds its filter graph for the new size and
	// starts its frame counter again from zero; the records must keep
	// coming, paired by PTS, at the fixed record size. Before the pairing
	// stopped relying on the counter, the reader waited for a counter that
	// never caught up while ffmpeg blocked on a full pipe, and the view
	// was gone for the rest of the session.
	ffmpeg := testmedia.FFmpeg(t)
	landscape := testmedia.H264UnitsSized(t, 320, 240, 20, 10)
	portrait := testmedia.H264UnitsSized(t, 240, 320, 20, 10)
	units := append(landscape, portrait...)
	r, w := newPipe()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	dec, err := New(ctx, w, Options{FFmpeg: ffmpeg, Width: 160, Height: 120, Codec: source.H264, Logf: t.Logf,
		LogLevel: "error"})
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	length := record.Size + record.FrameLength(160, 120)
	recs := r.records(length)
	for i, nalus := range units {
		au := &source.AccessUnit{Seq: uint32(i), NALUs: nalus, PTS: int64(i) * 3000, RTPTs: uint32(i) * 3000,
			Keyframe: i%10 == 0}
		if err := dec.Push(au); err != nil {
			t.Fatalf("push %d: %v", i, err)
		}
		time.Sleep(33 * time.Millisecond)
	}
	dec.Flush(10 * time.Second)
	var got []uint32
	deadline := time.After(5 * time.Second)
	for len(got) < len(units) {
		select {
		case rec := <-recs:
			h, err := record.Unmarshal(rec)
			if err != nil {
				t.Fatal(err)
			}
			if h.Width != 160 || h.Height != 120 {
				t.Fatalf("record of %dx%d", h.Width, h.Height)
			}
			got = append(got, h.Seq)
		case <-deadline:
			t.Fatalf("%d of %d frames after the picture turned: %v", len(got), len(units), got)
		}
	}
	for i, seq := range got {
		if seq != uint32(i) {
			t.Fatalf("frames out of order or mispaired: %v", got)
		}
	}
	if st := dec.Stats(); st.Unpaired != 0 || st.Skipped != 0 {
		t.Fatalf("stats %+v", st)
	}
}

func TestThreadsChangeOnlyTheThreadingArguments(t *testing.T) {
	one := Options{LogLevel: "error"}.args("F")
	if !slices.Equal(one, Options{LogLevel: "error", Threads: 1}.args("F")) {
		t.Fatalf("zero and one thread differ: %v", one)
	}
	at := slices.Index(one, "-flags")
	if at < 0 || !slices.Equal(one[at:at+4], []string{"-flags", "low_delay", "-threads", "1"}) {
		t.Fatalf("one thread: %v", one)
	}
	two := Options{LogLevel: "error", Threads: 2}.args("F")
	if slices.Contains(two, "low_delay") {
		t.Fatalf("frame threads with low_delay, which turns them off: %v", two)
	}
	at2 := slices.Index(two, "-threads")
	if at2 != at || !slices.Equal(two[at2:at2+4], []string{"-threads", "2", "-thread_type", "frame"}) {
		t.Fatalf("two threads: %v", two)
	}
	// Everything else is the same command.
	cut := func(args []string, at int) string {
		return strings.Join(slices.Delete(slices.Clone(args), at, at+4), " ")
	}
	if a, b := cut(one, at), cut(two, at2); a != b {
		t.Fatalf("%q\n%q", a, b)
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
	if d.stats.Skipped != 2 || len(d.pending) != 2 {
		t.Fatalf("skipped %d, pending %d", d.stats.Skipped, len(d.pending))
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
