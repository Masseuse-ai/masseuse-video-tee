package decode

import (
	"io"
	"sync"
)

// recordPipe is an io.Writer whose reader side hands out fixed-size
// records as they complete, for tests that check frames come out live.
type recordPipe struct {
	mu  sync.Mutex
	buf []byte
	ch  chan []byte
	n   int
}

func newPipe() (*recordPipe, io.Writer) {
	p := &recordPipe{ch: make(chan []byte, 64)}
	return p, p
}

func (p *recordPipe) Write(b []byte) (int, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.buf = append(p.buf, b...)
	for p.n > 0 && len(p.buf) >= p.n {
		rec := append([]byte(nil), p.buf[:p.n]...)
		p.buf = p.buf[p.n:]
		p.ch <- rec
	}
	return len(b), nil
}

// records sets the record size and returns the channel they arrive on.
func (p *recordPipe) records(n int) <-chan []byte {
	p.mu.Lock()
	p.n = n
	for len(p.buf) >= p.n {
		rec := append([]byte(nil), p.buf[:p.n]...)
		p.buf = p.buf[p.n:]
		p.ch <- rec
	}
	p.mu.Unlock()
	return p.ch
}
