// Package gate stands between a stream and its decoder: reading the stream
// never waits on the decoder, and the decoder is never given a unit that
// refers to one it was not given.
//
// Offered units go into a queue that one goroutine hands to the decoder in
// order. A unit is left out when it is damaged (source.AccessUnit.Damaged)
// or when the queue is full - its oldest unit has waited MaxWait, or it
// holds MaxUnits units or MaxBytes bytes - and from then on so is every
// unit until a keyframe that is not damaged arrives while the queue has
// room. What was queued before is still decoded: nothing it refers to was
// left out. A stall therefore costs frames, never a picture decoded from a
// reference that is missing.
package gate

import (
	"errors"
	"sync"
	"time"

	"github.com/Masseuse-ai/masseuse-video-tee/workload/reader/source"
)

// Defaults for Options' zero fields.
const (
	// DefaultMaxWait is how long the oldest queued unit may wait for the
	// decoder before the gate starts leaving units out: how far behind the
	// stream the decoder may fall, whatever the bit rate.
	DefaultMaxWait = time.Second
	// DefaultMaxUnits and DefaultMaxBytes bound the queue whatever the wait.
	DefaultMaxUnits = 256
	DefaultMaxBytes = 64 << 20
)

// ErrClosed is returned by Offer after Close.
var ErrClosed = errors.New("gate: closed")

// Options configure a Gate.
type Options struct {
	// MaxWait, MaxUnits and MaxBytes say when the queue is full; zero means
	// the default.
	MaxWait  time.Duration
	MaxUnits int
	MaxBytes int
	// Logf receives a line when the gate starts and stops leaving units
	// out; nil discards them.
	Logf func(format string, args ...any)
	// Now replaces the clock (tests).
	Now func() time.Time
}

// Stats counts what went through the gate.
type Stats struct {
	// Offered is how many units were offered, Dropped how many of them were
	// left out.
	Offered, Dropped uint64
	// Episodes is how many times the gate started leaving units out.
	Episodes uint64
	// Queued is how many units wait for the decoder now.
	Queued int
}

// Gate is the queue and the goroutine that empties it.
type Gate struct {
	opts Options

	mu       sync.Mutex
	cond     *sync.Cond
	queue    []entry
	bytes    int
	dropping bool
	reason   string // why the current episode began
	episode  uint64 // units left out in it
	started  bool
	closed   bool
	abandon  bool
	err      error
	stats    Stats
	done     chan struct{}
}

type entry struct {
	au   *source.AccessUnit
	at   time.Time
	size int
}

// New returns a Gate; its queue fills until Start.
func New(opts Options) *Gate {
	if opts.MaxWait <= 0 {
		opts.MaxWait = DefaultMaxWait
	}
	if opts.MaxUnits <= 0 {
		opts.MaxUnits = DefaultMaxUnits
	}
	if opts.MaxBytes <= 0 {
		opts.MaxBytes = DefaultMaxBytes
	}
	if opts.Logf == nil {
		opts.Logf = func(string, ...any) {}
	}
	if opts.Now == nil {
		opts.Now = time.Now
	}
	g := &Gate{opts: opts, done: make(chan struct{})}
	g.cond = sync.NewCond(&g.mu)
	return g
}

// Start hands the queued units to push, one at a time and in order, until
// Close. push may block; its first error ends the hand-over and is what
// Offer returns from then on.
func (g *Gate) Start(push func(*source.AccessUnit) error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.started {
		return
	}
	g.started = true
	go g.feed(push)
}

// Offer queues au for the decoder or leaves it out, without waiting on the
// decoder. It returns the decoder's error once there is one.
func (g *Gate) Offer(au *source.AccessUnit) error {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.err != nil {
		return g.err
	}
	if g.closed {
		return ErrClosed
	}
	g.stats.Offered++
	now := g.opts.Now()
	size := unitSize(au)
	full := g.full(now, size)
	switch {
	case au.Damaged:
		g.leaveOut(au, "a unit arrived damaged")
		return nil
	case g.dropping && (!au.Keyframe || full):
		g.leaveOut(au, "")
		return nil
	case full:
		g.leaveOut(au, "the decoder fell behind")
		return nil
	}
	if g.dropping {
		g.dropping = false
		g.opts.Logf("gate: resumed at keyframe %d after leaving out %d units (%s)", au.Seq, g.episode, g.reason)
	}
	g.queue = append(g.queue, entry{au: au, at: now, size: size})
	g.bytes += size
	g.cond.Signal()
	return nil
}

// Close stops taking units and lets the decoder have what is queued, for at
// most wait; past it, the hand-over stops after the unit it is on.
func (g *Gate) Close(wait time.Duration) {
	g.mu.Lock()
	if g.closed {
		g.mu.Unlock()
		return
	}
	g.closed = true
	started := g.started
	g.cond.Broadcast()
	g.mu.Unlock()
	if !started {
		return
	}
	select {
	case <-g.done:
	case <-time.After(wait):
		g.mu.Lock()
		g.abandon = true
		g.mu.Unlock()
		g.opts.Logf("gate: the decoder did not take the queue within %s", wait)
	}
}

// Stats is a snapshot of the counts.
func (g *Gate) Stats() Stats {
	g.mu.Lock()
	defer g.mu.Unlock()
	st := g.stats
	st.Queued = len(g.queue)
	return st
}

// full reports whether the queue has no room for a unit of size bytes: its
// oldest unit has waited MaxWait, or it holds MaxUnits units, or the unit
// would take it past MaxBytes. Caller holds g.mu.
func (g *Gate) full(now time.Time, size int) bool {
	if len(g.queue) == 0 {
		return false
	}
	return now.Sub(g.queue[0].at) >= g.opts.MaxWait ||
		len(g.queue) >= g.opts.MaxUnits ||
		g.bytes+size > g.opts.MaxBytes
}

// leaveOut drops au, opening an episode if this is its first unit. Caller
// holds g.mu.
func (g *Gate) leaveOut(au *source.AccessUnit, reason string) {
	g.stats.Dropped++
	if !g.dropping {
		g.dropping = true
		g.stats.Episodes++
		g.episode = 0
		g.reason = reason
		g.opts.Logf("gate: leaving out units from %d to the next whole keyframe: %s", au.Seq, reason)
	}
	g.episode++
}

func (g *Gate) feed(push func(*source.AccessUnit) error) {
	defer close(g.done)
	for {
		g.mu.Lock()
		for len(g.queue) == 0 && !g.closed {
			g.cond.Wait()
		}
		if len(g.queue) == 0 || g.abandon {
			g.mu.Unlock()
			return
		}
		e := g.queue[0]
		g.queue[0] = entry{}
		g.queue = g.queue[1:]
		g.bytes -= e.size
		g.mu.Unlock()
		if err := push(e.au); err != nil {
			g.mu.Lock()
			g.err = err
			g.queue, g.bytes = nil, 0
			g.mu.Unlock()
			return
		}
	}
}

func unitSize(au *source.AccessUnit) int {
	n := 0
	for _, nalu := range au.NALUs {
		n += len(nalu)
	}
	return n
}
