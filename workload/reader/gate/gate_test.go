package gate

import (
	"errors"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/Masseuse-ai/masseuse-video-tee/workload/reader/source"
)

func unit(seq int, keyframe bool) *source.AccessUnit {
	return &source.AccessUnit{Seq: uint32(seq), Keyframe: keyframe, NALUs: [][]byte{make([]byte, 100)}}
}

func damaged(seq int, keyframe bool) *source.AccessUnit {
	au := unit(seq, keyframe)
	au.Damaged = true
	return au
}

// clock is a time the test moves by hand.
type clock struct {
	mu sync.Mutex
	t  time.Time
}

func newClock() *clock { return &clock{t: time.Date(2026, 9, 30, 12, 0, 0, 0, time.UTC)} }

func (c *clock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *clock) set(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = time.Date(2026, 9, 30, 12, 0, 0, 0, time.UTC).Add(d)
}

// sink is a decoder that records what it is handed and, until released,
// holds on to the first unit it gets.
type sink struct {
	mu       sync.Mutex
	got      []uint32
	err      error
	arrived  chan struct{}
	released chan struct{}
	once     sync.Once
}

func newSink(held bool) *sink {
	s := &sink{arrived: make(chan struct{}, 1), released: make(chan struct{})}
	if !held {
		s.release()
	}
	return s
}

func (s *sink) push(au *source.AccessUnit) error {
	s.mu.Lock()
	s.got = append(s.got, au.Seq)
	err := s.err
	s.mu.Unlock()
	select {
	case s.arrived <- struct{}{}:
	default:
	}
	<-s.released
	return err
}

func (s *sink) release() { s.once.Do(func() { close(s.released) }) }

func (s *sink) seqs() []uint32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]uint32(nil), s.got...)
}

// waitFor waits until the sink has been handed n units.
func (s *sink) waitFor(t *testing.T, n int) {
	t.Helper()
	deadline := time.After(5 * time.Second)
	for len(s.seqs()) < n {
		select {
		case <-s.arrived:
		case <-deadline:
			t.Fatalf("the sink got %v, not %d units", s.seqs(), n)
		}
	}
}

func offer(t *testing.T, g *Gate, units ...*source.AccessUnit) {
	t.Helper()
	for _, au := range units {
		if err := g.Offer(au); err != nil {
			t.Fatalf("offer %d: %v", au.Seq, err)
		}
	}
}

func TestUnitsReachTheDecoderInOrder(t *testing.T) {
	g := New(Options{})
	s := newSink(false)
	g.Start(s.push)
	var want []uint32
	for i := 0; i < 50; i++ {
		offer(t, g, unit(i, i%10 == 0))
		want = append(want, uint32(i))
	}
	g.Close(5 * time.Second)
	if got := s.seqs(); !slices.Equal(got, want) {
		t.Fatalf("the decoder got %v", got)
	}
	if st := g.Stats(); st.Offered != 50 || st.Dropped != 0 || st.Episodes != 0 || st.Queued != 0 {
		t.Fatalf("stats %+v", st)
	}
}

func TestADecoderBehindLosesTheUnitsUpToAKeyframe(t *testing.T) {
	clk := newClock()
	g := New(Options{MaxWait: time.Second, Now: clk.Now})
	s := newSink(true)
	g.Start(s.push)
	offer(t, g, unit(0, true))
	s.waitFor(t, 1) // the decoder is on unit 0 and stays there
	offer(t, g, unit(1, false))
	clk.set(500 * time.Millisecond)
	offer(t, g, unit(2, false))
	// Unit 1 has waited a second: the queue is full.
	clk.set(time.Second)
	offer(t, g, unit(3, false), unit(4, false))
	// A keyframe while the queue is still full does not end it.
	clk.set(1200 * time.Millisecond)
	offer(t, g, unit(5, true))
	if st := g.Stats(); st.Dropped != 3 || st.Episodes != 1 || st.Queued != 2 {
		t.Fatalf("stats while behind %+v", st)
	}
	// The decoder catches up on what was queued, which refers to nothing
	// left out.
	s.release()
	s.waitFor(t, 3)
	clk.set(1300 * time.Millisecond)
	offer(t, g, unit(6, false)) // still waiting for a keyframe
	offer(t, g, unit(7, true))  // the queue has room: resumed
	offer(t, g, unit(8, false))
	g.Close(5 * time.Second)
	if got := s.seqs(); !slices.Equal(got, []uint32{0, 1, 2, 7, 8}) {
		t.Fatalf("the decoder got %v", got)
	}
	if st := g.Stats(); st.Offered != 9 || st.Dropped != 4 || st.Episodes != 1 || st.Queued != 0 {
		t.Fatalf("stats %+v", st)
	}
}

func TestADamagedUnitIsLeftOutWithWhatRefersToIt(t *testing.T) {
	g := New(Options{})
	s := newSink(false)
	g.Start(s.push)
	offer(t, g,
		unit(0, true), unit(1, false),
		damaged(2, false), unit(3, false), unit(4, false),
		damaged(5, true), // a damaged keyframe is no place to resume
		unit(6, false),
		unit(7, true), unit(8, false),
		damaged(9, false), unit(10, true), unit(11, false))
	g.Close(5 * time.Second)
	if got := s.seqs(); !slices.Equal(got, []uint32{0, 1, 7, 8, 10, 11}) {
		t.Fatalf("the decoder got %v", got)
	}
	if st := g.Stats(); st.Offered != 12 || st.Dropped != 6 || st.Episodes != 2 {
		t.Fatalf("stats %+v", st)
	}
}

func TestTheQueueIsBoundedByUnitsAndBytes(t *testing.T) {
	cases := map[string]Options{
		"units": {MaxUnits: 3},
		"bytes": {MaxBytes: 350}, // three 100-byte units fit, a fourth does not
	}
	for name, opts := range cases {
		t.Run(name, func(t *testing.T) {
			g := New(opts)
			s := newSink(true)
			g.Start(s.push)
			offer(t, g, unit(0, true))
			s.waitFor(t, 1)
			offer(t, g, unit(1, false), unit(2, false), unit(3, false), unit(4, false))
			if st := g.Stats(); st.Queued != 3 || st.Dropped != 1 {
				t.Fatalf("stats %+v", st)
			}
			s.release()
			g.Close(5 * time.Second)
			if got := s.seqs(); !slices.Equal(got, []uint32{0, 1, 2, 3}) {
				t.Fatalf("the decoder got %v", got)
			}
		})
	}
}

func TestTheDecodersErrorComesBackFromOffer(t *testing.T) {
	g := New(Options{})
	s := newSink(false)
	boom := errors.New("boom")
	s.err = boom
	g.Start(s.push)
	offer(t, g, unit(0, true))
	deadline := time.Now().Add(5 * time.Second)
	for {
		err := g.Offer(unit(1, false))
		if errors.Is(err, boom) {
			break
		}
		if err != nil || time.Now().After(deadline) {
			t.Fatalf("offer after the decoder failed: %v", err)
		}
		time.Sleep(time.Millisecond)
	}
}

func TestCloseLetsTheDecoderHaveTheQueue(t *testing.T) {
	g := New(Options{})
	s := newSink(true)
	g.Start(s.push)
	offer(t, g, unit(0, true), unit(1, false), unit(2, false))
	go func() {
		time.Sleep(50 * time.Millisecond)
		s.release()
	}()
	g.Close(5 * time.Second)
	if got := s.seqs(); !slices.Equal(got, []uint32{0, 1, 2}) {
		t.Fatalf("the decoder got %v", got)
	}
	if err := g.Offer(unit(3, false)); !errors.Is(err, ErrClosed) {
		t.Fatalf("offer after close: %v", err)
	}
}

func TestCloseDoesNotWaitOnAStuckDecoder(t *testing.T) {
	g := New(Options{})
	s := newSink(true)
	g.Start(s.push)
	offer(t, g, unit(0, true), unit(1, false), unit(2, false))
	s.waitFor(t, 1)
	started := time.Now()
	g.Close(100 * time.Millisecond)
	if d := time.Since(started); d < 100*time.Millisecond || d > 3*time.Second {
		t.Fatalf("close took %s", d)
	}
	// Let go, the hand-over stops after the unit it was on.
	s.release()
	<-g.done
	if got := s.seqs(); !slices.Equal(got, []uint32{0}) {
		t.Fatalf("the decoder got %v", got)
	}
}
