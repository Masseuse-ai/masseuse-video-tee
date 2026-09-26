// Package source reads the first video track of an RTSP stream into access
// units that keep their time: the RTP timestamp, the presentation
// timestamp derived from it, and the absolute time the sender's RTCP sender
// reports give the unit - the sender's own clock, which is what lines two
// cameras up.
package source

import (
	"context"
	"errors"
	"fmt"
	"net"
	"sync"
	"time"

	"github.com/bluenviron/gortsplib/v5"
	"github.com/bluenviron/gortsplib/v5/pkg/base"
	"github.com/bluenviron/gortsplib/v5/pkg/description"
	"github.com/bluenviron/gortsplib/v5/pkg/format"
	"github.com/bluenviron/mediacommon/v2/pkg/codecs/h264"
	"github.com/bluenviron/mediacommon/v2/pkg/codecs/h265"
	"github.com/pion/rtp"
)

// Codec is the video track's codec.
type Codec uint8

// Codecs, numbered as record.Codec* are.
const (
	H264 Codec = 1
	H265 Codec = 2
)

// String implements fmt.Stringer.
func (c Codec) String() string {
	switch c {
	case H264:
		return "h264"
	case H265:
		return "h265"
	}
	return fmt.Sprintf("codec(%d)", uint8(c))
}

// AccessUnit is one coded picture with its time.
type AccessUnit struct {
	// Seq numbers the units this source produced, from 0.
	Seq uint32
	// NALUs is the unit's NAL units, without start codes. A random-access
	// unit carries the track's parameter sets first (from the SDP when the
	// sender did not put them in-band), so a decoder can start on it.
	NALUs [][]byte
	// PTS is the presentation timestamp in RTP clock ticks (90 kHz),
	// unwrapped and counted from the first unit.
	PTS int64
	// RTPTs is the unit's RTP timestamp as sent.
	RTPTs uint32
	// NTP is the moment the unit belongs to on the sender's clock, from its
	// RTCP sender reports; valid when NTPValid.
	NTP      time.Time
	NTPValid bool
	// Keyframe is whether the unit is a random-access point.
	Keyframe bool
}

// Default limits.
const (
	// HoldForNTP bounds how long units are held back waiting for the
	// sender's first report, and HoldUnits how many. The relay reports
	// right after the first packet a reader gets, so the hold is normally
	// over within one unit; a sender that never reports flows arrival-timed
	// after it.
	HoldForNTP = time.Second
	HoldUnits  = 64
	// DefaultReadTimeout is how long the connection may stay silent.
	DefaultReadTimeout = 10 * time.Second
)

// ErrNoVideo is returned by Open when the stream has no H.264 or H.265
// video track.
var ErrNoVideo = errors.New("source: no H.264 or H.265 video track")

// Options configure Open.
type Options struct {
	// ReadTimeout is the RTSP read timeout; 0 means DefaultReadTimeout.
	ReadTimeout time.Duration
	// Dial replaces the network dial (tests); nil means net.Dialer.
	Dial func(ctx context.Context, network, address string) (net.Conn, error)
	// Hold overrides HoldForNTP; 0 means the default.
	Hold time.Duration
	// Logf receives one line per notable event; nil discards them.
	Logf func(format string, args ...any)
}

// Source is one connection to a stream's video track.
type Source struct {
	c     *gortsplib.Client
	desc  *description.Session
	medi  *description.Media
	forma format.Format
	codec Codec
	opts  Options

	decode func(*rtp.Packet) ([][]byte, error)
	params [][]byte // parameter sets from the SDP, in decoding order

	mu      sync.Mutex
	seq     uint32
	started bool // a keyframe has been seen; units flow from it
	settled bool // the sender has reported, or the hold ran out
	held    []*AccessUnit
	since   time.Time
	emit    func(*AccessUnit) error
	err     error
}

// Open connects to rawURL, describes the stream and sets up its first
// H.264 or H.265 video track. Run starts the packets.
func Open(ctx context.Context, rawURL string, opts Options) (*Source, error) {
	u, err := base.ParseURL(rawURL)
	if err != nil {
		return nil, fmt.Errorf("source: %w", err)
	}
	if opts.ReadTimeout == 0 {
		opts.ReadTimeout = DefaultReadTimeout
	}
	if opts.Hold == 0 {
		opts.Hold = HoldForNTP
	}
	if opts.Logf == nil {
		opts.Logf = func(string, ...any) {}
	}
	tcp := gortsplib.ProtocolTCP
	c := &gortsplib.Client{
		Scheme:      u.Scheme,
		Host:        u.Host,
		Protocol:    &tcp,
		ReadTimeout: opts.ReadTimeout,
		DialContext: opts.Dial,
	}
	if err := c.Start(); err != nil {
		return nil, fmt.Errorf("source: %w", err)
	}
	s := &Source{c: c, opts: opts}
	if err := s.setup(ctx, u); err != nil {
		c.Close()
		return nil, err
	}
	return s, nil
}

func (s *Source) setup(ctx context.Context, u *base.URL) error {
	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case <-ctx.Done():
			s.c.Close()
		case <-done:
		}
	}()
	desc, _, err := s.c.Describe(u)
	if err != nil {
		return fmt.Errorf("source: describe: %w", err)
	}
	s.desc = desc
	for _, medi := range desc.Medias {
		if medi.Type != description.MediaTypeVideo {
			continue
		}
		for _, forma := range medi.Formats {
			switch v := forma.(type) {
			case *format.H264:
				d, err := v.CreateDecoder()
				if err != nil {
					return fmt.Errorf("source: %w", err)
				}
				s.medi, s.forma, s.codec, s.decode = medi, forma, H264, d.Decode
				if v.SPS != nil && v.PPS != nil {
					s.params = [][]byte{v.SPS, v.PPS}
				}
			case *format.H265:
				d, err := v.CreateDecoder()
				if err != nil {
					return fmt.Errorf("source: %w", err)
				}
				s.medi, s.forma, s.codec, s.decode = medi, forma, H265, d.Decode
				if v.VPS != nil && v.SPS != nil && v.PPS != nil {
					s.params = [][]byte{v.VPS, v.SPS, v.PPS}
				}
			default:
				continue
			}
			break
		}
		if s.medi != nil {
			break
		}
	}
	if s.medi == nil {
		return ErrNoVideo
	}
	if _, err := s.c.Setup(desc.BaseURL, s.medi, 0, 0); err != nil {
		return fmt.Errorf("source: setup: %w", err)
	}
	return nil
}

// Codec is the track's codec.
func (s *Source) Codec() Codec { return s.codec }

// Size is the picture size the SDP's parameter sets declare, or zeros when
// they are absent.
func (s *Source) Size() (width, height int) {
	switch v := s.forma.(type) {
	case *format.H264:
		var sps h264.SPS
		if v.SPS != nil && sps.Unmarshal(v.SPS) == nil {
			return sps.Width(), sps.Height()
		}
	case *format.H265:
		var sps h265.SPS
		if v.SPS != nil && sps.Unmarshal(v.SPS) == nil {
			return sps.Width(), sps.Height()
		}
	}
	return 0, 0
}

// Run plays the track and hands every access unit to emit, in order, until
// the connection ends, ctx is done, or emit returns an error - which Run
// then returns. Units before the first keyframe are dropped: nothing can
// decode them. Units are held back while the sender's time is unknown (see
// HoldForNTP) and released timed once it is known, so that the first unit
// out already carries the sender's clock when the sender reports promptly.
func (s *Source) Run(ctx context.Context, emit func(*AccessUnit) error) error {
	s.mu.Lock()
	s.emit = emit
	s.mu.Unlock()
	s.c.OnPacketRTP(s.medi, s.forma, s.onPacket)
	if _, err := s.c.Play(nil); err != nil {
		return fmt.Errorf("source: play: %w", err)
	}
	waitErr := make(chan error, 1)
	go func() { waitErr <- s.c.Wait() }()
	var err error
	select {
	case err = <-waitErr:
	case <-ctx.Done():
		s.c.Close()
		<-waitErr
		err = ctx.Err()
	}
	s.mu.Lock()
	if s.err != nil {
		err = s.err
	}
	// Whatever was still held goes out arrival-timed rather than lost.
	held := s.held
	s.held = nil
	s.mu.Unlock()
	for _, au := range held {
		if e := emit(au); e != nil && err == nil {
			err = e
		}
	}
	return err
}

// Close ends the connection; Run returns.
func (s *Source) Close() { s.c.Close() }

func (s *Source) onPacket(pkt *rtp.Packet) {
	pts, ok := s.c.PacketPTS(s.medi, pkt)
	if !ok {
		return // before the first packet whose timestamp is trustworthy
	}
	nalus, err := s.decode(pkt)
	if err != nil {
		// More packets needed, or a fragment whose start was lost.
		return
	}
	keyframe := s.isKeyframe(nalus)
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err != nil {
		return
	}
	if !s.started {
		if !keyframe {
			return
		}
		s.started = true
	}
	if keyframe && len(s.params) > 0 && !s.hasParams(nalus) {
		nalus = append(append([][]byte{}, s.params...), nalus...)
	}
	au := &AccessUnit{Seq: s.seq, NALUs: nalus, PTS: pts, RTPTs: pkt.Timestamp, Keyframe: keyframe}
	s.seq++
	au.NTP, au.NTPValid = s.c.PacketNTP(s.medi, pkt)
	if !au.NTPValid {
		au.NTP = time.Time{}
	}
	if s.settled {
		s.deliver(au)
		return
	}
	if au.NTPValid {
		// The sender has reported: its time is known for every timestamp,
		// the held units' included.
		s.settled = true
		for _, h := range s.held {
			h.NTP, h.NTPValid = s.c.PacketNTP(s.medi, &rtp.Packet{Header: rtp.Header{PayloadType: pkt.PayloadType, Timestamp: h.RTPTs}})
			s.deliver(h)
		}
		s.held = nil
		s.deliver(au)
		return
	}
	now := time.Now()
	if len(s.held) == 0 {
		s.since = now
	}
	s.held = append(s.held, au)
	if now.Sub(s.since) < s.opts.Hold && len(s.held) < HoldUnits {
		return
	}
	// The sender is not saying: the units flow timed by their arrival, and
	// the producer paces them by their RTP timestamps.
	s.settled = true
	s.opts.Logf("source: no sender report within %s; frames are arrival-timed until one comes", s.opts.Hold)
	for _, h := range s.held {
		s.deliver(h)
	}
	s.held = nil
}

// deliver hands au to emit; the first error stops the source. Caller holds
// s.mu.
func (s *Source) deliver(au *AccessUnit) {
	if s.err != nil {
		return
	}
	if err := s.emit(au); err != nil {
		s.err = err
		go s.c.Close()
	}
}

func (s *Source) isKeyframe(nalus [][]byte) bool {
	if s.codec == H265 {
		return h265.IsRandomAccess(nalus)
	}
	return h264.IsRandomAccess(nalus)
}

// hasParams reports whether the unit carries its own parameter sets.
func (s *Source) hasParams(nalus [][]byte) bool {
	for _, n := range nalus {
		if len(n) == 0 {
			continue
		}
		if s.codec == H265 {
			switch h265.NALUType((n[0] >> 1) & 0x3F) {
			case h265.NALUType_VPS_NUT, h265.NALUType_SPS_NUT, h265.NALUType_PPS_NUT:
				return true
			}
		} else if h264.NALUType(n[0]&0x1F) == h264.NALUTypeSPS {
			return true
		}
	}
	return false
}
