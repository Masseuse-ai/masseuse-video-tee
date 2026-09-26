// Package decode turns access units into frame records through an ffmpeg
// child, without losing which frame came from which unit.
//
// ffmpeg is fed MPEG-TS on its stdin, each unit stamped with its PTS, and
// asked for raw yuv420p frames of a fixed size on stdout with `-copyts
// -fps_mode passthrough`, so it neither drops nor duplicates frames and
// keeps their timestamps. A `metadata=print` filter writes one line per
// frame on a third pipe with the frame's PTS as the decoder saw it; the
// frame's record is then the unit with that PTS - its RTP timestamp and
// the sender's time - which the frame reader prepends as a record.Header.
package decode

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/bluenviron/mediacommon/v2/pkg/codecs/h264"
	"github.com/bluenviron/mediacommon/v2/pkg/codecs/h265"
	"github.com/bluenviron/mediacommon/v2/pkg/formats/mpegts"
	mpegtscodecs "github.com/bluenviron/mediacommon/v2/pkg/formats/mpegts/codecs"

	"github.com/FemLed/masseuse-video-tee/workload/reader/record"
	"github.com/FemLed/masseuse-video-tee/workload/reader/source"
)

const (
	// ptsBase is added to every PTS handed to ffmpeg: the first unit's PTS
	// is 0 and a stream with B-frames has DTS below PTS, which MPEG-TS
	// cannot carry below zero.
	ptsBase = int64(90000 * 60)
	// ptsMask keeps timestamps within MPEG-TS's 33 bits.
	ptsMask = int64(1)<<33 - 1
	// PendingCap bounds the units waiting for their frame: past it the
	// oldest are given up (the decoder dropped them).
	PendingCap = 300
)

// Options configure a Decoder.
type Options struct {
	// FFmpeg is the ffmpeg executable; "" means "ffmpeg" on PATH.
	FFmpeg string
	// Width and Height are the frame size every record carries; the
	// picture is scaled to fit and padded to it.
	Width, Height int
	// Codec is the units' codec.
	Codec source.Codec
	// Logf receives one line per notable event; nil discards them.
	Logf func(format string, args ...any)
	// Stderr receives ffmpeg's stderr; nil means the process's.
	Stderr io.Writer
	// LogLevel is ffmpeg's -loglevel; "" means "error".
	LogLevel string
}

// Stats counts what happened to the units and frames.
type Stats struct {
	Units, Frames, Unpaired, Dropped uint64
}

// Decoder is one ffmpeg child and the pairing around it.
type Decoder struct {
	opts Options
	out  io.Writer

	cmd   *exec.Cmd
	stdin io.WriteCloser
	meta  *os.File // our end of the metadata pipe
	mux   *mpegts.Writer
	track *mpegts.Track
	dts   func([][]byte, int64) (int64, error)

	mu      sync.Mutex
	pending []*pendingUnit
	stats   Stats
	dtsWarn bool

	lines chan frameLine
	done  chan struct{}
	err   error
	once  sync.Once
}

type pendingUnit struct {
	au  *source.AccessUnit
	pts int64 // as handed to ffmpeg
}

type frameLine struct {
	index int64
	pts   int64
}

// New starts ffmpeg; records go to out.
func New(ctx context.Context, out io.Writer, opts Options) (*Decoder, error) {
	if opts.Width <= 0 || opts.Height <= 0 || opts.Width%2 != 0 || opts.Height%2 != 0 {
		return nil, fmt.Errorf("decode: frame size %dx%d must be even and positive", opts.Width, opts.Height)
	}
	if opts.FFmpeg == "" {
		opts.FFmpeg = "ffmpeg"
	}
	if opts.Logf == nil {
		opts.Logf = func(string, ...any) {}
	}
	if opts.Stderr == nil {
		opts.Stderr = os.Stderr
	}
	if opts.LogLevel == "" {
		opts.LogLevel = "error"
	}
	d := &Decoder{opts: opts, out: out, lines: make(chan frameLine, 64), done: make(chan struct{})}
	switch opts.Codec {
	case source.H264:
		d.track = &mpegts.Track{Codec: &mpegtscodecs.H264{}}
		x := h264.NewDTSExtractor()
		d.dts = x.Extract
	case source.H265:
		d.track = &mpegts.Track{Codec: &mpegtscodecs.H265{}}
		x := h265.NewDTSExtractor()
		d.dts = x.Extract
	default:
		return nil, fmt.Errorf("decode: codec %s", opts.Codec)
	}

	metaR, metaW, err := os.Pipe()
	if err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	d.meta = metaR
	// `print` writes a frame line only for frames that carry metadata, so
	// `add` gives each a key first; `direct` bypasses the output buffer,
	// without which the lines would arrive kilobytes late. The pipe URL's
	// colon is escaped twice: once for the graph parser, once for the
	// filter's own option parser.
	filters := fmt.Sprintf("scale=%d:%d:force_original_aspect_ratio=decrease:force_divisible_by=2,"+
		"pad=%d:%d:-1:-1,metadata=mode=add:key=r:value=1,"+
		"metadata=mode=print:file=pipe\\\\:3:direct=1",
		opts.Width, opts.Height, opts.Width, opts.Height)
	// The probe ends with the second unit (`-analyzeduration 1`: 0 would
	// mean the default five seconds); the first, a keyframe carrying its
	// parameter sets, told ffmpeg what it needs, and the probe's packets
	// are replayed, so nothing is lost to it (`-fflags nobuffer` would
	// discard them). One decoding thread: frame threads each add a frame
	// of delay.
	d.cmd = exec.CommandContext(ctx, opts.FFmpeg,
		"-nostdin", "-hide_banner", "-loglevel", opts.LogLevel,
		"-flags", "low_delay", "-threads", "1",
		"-analyzeduration", "1", "-probesize", "65536",
		"-f", "mpegts", "-i", "pipe:0",
		"-copyts", "-vf", filters, "-fps_mode", "passthrough",
		"-f", "rawvideo", "-pix_fmt", "yuv420p", "-flush_packets", "1", "pipe:1")
	d.cmd.Stderr = opts.Stderr
	d.cmd.ExtraFiles = []*os.File{metaW}
	stdin, err := d.cmd.StdinPipe()
	if err != nil {
		metaR.Close()
		metaW.Close()
		return nil, fmt.Errorf("decode: %w", err)
	}
	stdout, err := d.cmd.StdoutPipe()
	if err != nil {
		metaR.Close()
		metaW.Close()
		return nil, fmt.Errorf("decode: %w", err)
	}
	if err := d.cmd.Start(); err != nil {
		metaR.Close()
		metaW.Close()
		return nil, fmt.Errorf("decode: ffmpeg: %w", err)
	}
	metaW.Close() // the child holds its end
	d.stdin = stdin
	d.mux = &mpegts.Writer{W: stdin, Tracks: []*mpegts.Track{d.track}}
	if err := d.mux.Initialize(); err != nil {
		d.Close()
		return nil, fmt.Errorf("decode: %w", err)
	}
	go d.readLines(metaR)
	go d.readFrames(stdout)
	return d, nil
}

// Push hands a unit to ffmpeg.
func (d *Decoder) Push(au *source.AccessUnit) error {
	select {
	case <-d.done:
		return d.Err()
	default:
	}
	pts := (au.PTS + ptsBase) & ptsMask
	dts, err := d.dts(au.NALUs, pts)
	if err != nil {
		d.mu.Lock()
		if !d.dtsWarn {
			d.dtsWarn = true
			d.opts.Logf("decode: cannot tell DTS from PTS (%v); using PTS", err)
		}
		d.mu.Unlock()
		dts = pts
	}
	d.mu.Lock()
	d.stats.Units++
	d.pending = append(d.pending, &pendingUnit{au: au, pts: pts})
	if len(d.pending) > PendingCap {
		d.stats.Dropped += uint64(len(d.pending) - PendingCap)
		d.pending = d.pending[len(d.pending)-PendingCap:]
	}
	d.mu.Unlock()
	if d.opts.Codec == source.H265 {
		err = d.mux.WriteH265(d.track, pts, dts, au.NALUs)
	} else {
		err = d.mux.WriteH264(d.track, pts, dts, au.NALUs)
	}
	if err != nil {
		return fmt.Errorf("decode: %w", err)
	}
	return nil
}

// readLines parses the metadata pipe: "frame:N pts:P pts_time:T" lines,
// each followed by the key we added.
func (d *Decoder) readLines(r io.ReadCloser) {
	defer r.Close()
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "frame:") {
			continue
		}
		fl, ok := parseFrameLine(line)
		if !ok {
			continue
		}
		select {
		case d.lines <- fl:
		case <-d.done:
			return
		}
	}
	close(d.lines)
}

// parseFrameLine reads "frame:12   pts:129000  pts_time:1.433333".
func parseFrameLine(line string) (frameLine, bool) {
	var fl frameLine
	var haveIndex, havePTS bool
	for _, field := range strings.Fields(line) {
		key, value, ok := strings.Cut(field, ":")
		if !ok {
			continue
		}
		switch key {
		case "frame":
			n, err := strconv.ParseInt(value, 10, 64)
			if err != nil {
				return fl, false
			}
			fl.index, haveIndex = n, true
		case "pts":
			n, err := strconv.ParseInt(value, 10, 64)
			if err != nil {
				return fl, false // NOPTS
			}
			fl.pts, havePTS = n, true
		}
	}
	return fl, haveIndex && havePTS
}

// readFrames reads fixed-size frames from ffmpeg's stdout and writes each
// as a record with the unit its metadata line names. Frames and lines are
// one to one: every frame passes the print filter once. The line's own
// frame counter is not relied on - when the sender's picture changes size
// (a phone turned), ffmpeg rebuilds the filter graph and the counter
// starts again from zero, while the PTS carries on.
func (d *Decoder) readFrames(r io.Reader) {
	defer d.finish()
	length := record.FrameLength(d.opts.Width, d.opts.Height)
	buf := make([]byte, record.Size+length)
	br := bufio.NewReaderSize(r, 1<<20)
	for {
		if _, err := io.ReadFull(br, buf[record.Size:]); err != nil {
			if !errors.Is(err, io.EOF) && !errors.Is(err, io.ErrUnexpectedEOF) {
				d.fail(fmt.Errorf("decode: reading frames: %w", err))
			}
			return
		}
		line, ok := <-d.lines
		if !ok {
			d.fail(errors.New("decode: metadata pipe closed before its frame"))
			return
		}
		au := d.pair(line.pts)
		h := record.Header{
			Codec:  uint8(d.opts.Codec),
			Width:  uint16(d.opts.Width),
			Height: uint16(d.opts.Height),
			Length: uint32(length),
		}
		if au != nil {
			h.Seq = au.Seq
			h.RTPTs = au.RTPTs
			if au.NTPValid {
				h.Flags |= record.FlagNTPValid
				h.NTPNs = au.NTP.UnixNano()
			}
			if au.Keyframe {
				h.Flags |= record.FlagKeyframe
			}
		}
		h.Marshal(buf[:record.Size])
		if au != nil {
			if _, err := d.out.Write(buf); err != nil {
				d.fail(fmt.Errorf("decode: writing records: %w", err))
				return
			}
			d.mu.Lock()
			d.stats.Frames++
			d.mu.Unlock()
		}
	}
}

// pair takes the pending unit with pts, dropping the ones before it (the
// decoder never gave them a frame). Without a match the oldest pending unit
// stands in and is counted as unpaired.
func (d *Decoder) pair(pts int64) *source.AccessUnit {
	d.mu.Lock()
	defer d.mu.Unlock()
	for i, p := range d.pending {
		if p.pts == pts {
			d.stats.Dropped += uint64(i)
			d.pending = append(d.pending[:0], d.pending[i+1:]...)
			return p.au
		}
	}
	d.stats.Unpaired++
	if len(d.pending) == 0 {
		return nil
	}
	p := d.pending[0]
	d.pending = append(d.pending[:0], d.pending[1:]...)
	return p.au
}

func (d *Decoder) fail(err error) {
	d.mu.Lock()
	if d.err == nil {
		d.err = err
	}
	d.mu.Unlock()
}

func (d *Decoder) finish() {
	d.once.Do(func() { close(d.done) })
}

// Err is why the decoder stopped, if it has.
func (d *Decoder) Err() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.err
}

// Done is closed when the frame stream has ended.
func (d *Decoder) Done() <-chan struct{} { return d.done }

// Stats is a snapshot of the counts.
func (d *Decoder) Stats() Stats {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.stats
}

// Flush ends the input: ffmpeg decodes what it has and exits, the frames
// come out, Done closes. Wait bounds the flush.
func (d *Decoder) Flush(wait time.Duration) {
	d.stdin.Close()
	select {
	case <-d.done:
	case <-time.After(wait):
		d.opts.Logf("decode: ffmpeg did not finish within %s", wait)
	}
}

// Close ends ffmpeg and releases the pipes.
func (d *Decoder) Close() error {
	d.stdin.Close()
	if d.cmd.Process != nil {
		_ = d.cmd.Process.Kill()
	}
	err := d.cmd.Wait()
	d.meta.Close()
	d.finish()
	var exit *exec.ExitError
	if errors.As(err, &exit) {
		return nil // killed, or ffmpeg's own exit: the frames say enough
	}
	return err
}
