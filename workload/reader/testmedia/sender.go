package testmedia

import (
	"sync"
	"testing"
	"time"

	"github.com/bluenviron/gortsplib/v5"
	"github.com/bluenviron/gortsplib/v5/pkg/base"
	"github.com/bluenviron/gortsplib/v5/pkg/description"
	"github.com/bluenviron/gortsplib/v5/pkg/format"
	"github.com/bluenviron/gortsplib/v5/pkg/format/rtph264"
	"github.com/bluenviron/mediacommon/v2/pkg/codecs/h264"
)

// FirstRTPTs is the RTP timestamp of the Sender's first unit.
const FirstRTPTs = 100000

// Sender is an RTSP server over a PipeNet that plays H.264 units in a
// loop, one every frame, each timed on its own clock: Base plus the
// unit's place in the media timeline (UnitTime).
type Sender struct {
	Net    *PipeNet
	Srv    *gortsplib.Server
	Stream *gortsplib.ServerStream
	Desc   *description.Session
	Media  *description.Media
	Base   time.Time
	// SDPFrom is the unit whose parameter sets the SDP carries; nil means
	// the first unit. Set from `configure`.
	SDPFrom [][]byte
	// AudioOnly describes a stream without video. Set from `configure`.
	AudioOnly bool

	units [][][]byte
	stop  chan struct{}
	wg    sync.WaitGroup
}

// StartSender starts a Sender; configure, if given, adjusts it and its
// server before they start. It is stopped when the test ends.
func StartSender(t *testing.T, units [][][]byte, configure func(*Sender, *gortsplib.Server)) *Sender {
	t.Helper()
	s := &Sender{Net: NewPipeNet(), units: units, stop: make(chan struct{}), Base: time.Now().Add(-2 * time.Second)}
	s.Srv = &gortsplib.Server{Handler: s, RTSPAddress: "127.0.0.1:8554", Listen: s.Net.Listen}
	if configure != nil {
		configure(s, s.Srv)
	}
	forma := &format.H264{PayloadTyp: 96, PacketizationMode: 1}
	if s.SDPFrom == nil && len(units) > 0 {
		s.SDPFrom = units[0]
	}
	for _, n := range s.SDPFrom {
		switch h264.NALUType(n[0] & 0x1F) {
		case h264.NALUTypeSPS:
			forma.SPS = n
		case h264.NALUTypePPS:
			forma.PPS = n
		}
	}
	s.Media = &description.Media{Type: description.MediaTypeVideo, Formats: []format.Format{forma}}
	s.Desc = &description.Session{Medias: []*description.Media{s.Media}}
	if s.AudioOnly {
		s.Desc = &description.Session{Medias: []*description.Media{
			{Type: description.MediaTypeAudio, Formats: []format.Format{&format.Opus{PayloadTyp: 97, ChannelCount: 1}}},
		}}
	}
	if err := s.Srv.Start(); err != nil {
		t.Fatal(err)
	}
	s.Stream = &gortsplib.ServerStream{Server: s.Srv, Desc: s.Desc}
	if err := s.Stream.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !s.AudioOnly {
		s.wg.Add(1)
		go s.play()
	}
	t.Cleanup(func() {
		close(s.stop)
		s.wg.Wait()
		s.Stream.Close()
		s.Srv.Close()
		s.Net.Close()
	})
	return s
}

// URL is what a client reads.
func (s *Sender) URL() string { return "rtsp://127.0.0.1:8554/cam" }

func (s *Sender) play() {
	defer s.wg.Done()
	enc := &rtph264.Encoder{PayloadType: 96, PacketizationMode: 1, PayloadMaxSize: 1200}
	if err := enc.Init(); err != nil {
		panic(err)
	}
	tick := time.NewTicker(33 * time.Millisecond)
	defer tick.Stop()
	for k := 0; ; k++ {
		select {
		case <-s.stop:
			return
		case <-tick.C:
		}
		pkts, err := enc.Encode(s.units[k%len(s.units)])
		if err != nil {
			continue
		}
		ts := uint32(FirstRTPTs + k*3000)
		ntp := s.Base.Add(time.Duration(k) * time.Second / 30)
		for _, p := range pkts {
			p.Timestamp = ts
			_ = s.Stream.WritePacketRTPWithNTP(s.Media, p, ntp)
		}
	}
}

// UnitTime is the moment the sender gave the unit with RTP timestamp ts.
func (s *Sender) UnitTime(ts uint32) time.Time {
	return s.Base.Add(time.Duration(int64(ts)-FirstRTPTs) * time.Second / 90000)
}

// OnDescribe implements gortsplib.ServerHandler.
func (s *Sender) OnDescribe(*gortsplib.ServerHandlerOnDescribeCtx) (*base.Response, *gortsplib.ServerStream, error) {
	return &base.Response{StatusCode: base.StatusOK}, s.Stream, nil
}

// OnSetup implements gortsplib.ServerHandler.
func (s *Sender) OnSetup(*gortsplib.ServerHandlerOnSetupCtx) (*base.Response, *gortsplib.ServerStream, error) {
	return &base.Response{StatusCode: base.StatusOK}, s.Stream, nil
}

// OnPlay implements gortsplib.ServerHandler.
func (s *Sender) OnPlay(*gortsplib.ServerHandlerOnPlayCtx) (*base.Response, error) {
	return &base.Response{StatusCode: base.StatusOK}, nil
}
