package record

import (
	"errors"
	"math"
	"testing"
)

func TestHeaderRoundTrip(t *testing.T) {
	h := Header{
		Flags:   FlagNTPValid | FlagKeyframe,
		Codec:   CodecH264,
		Seq:     41,
		NTPNs:   1_757_500_000_123_456_789,
		RTPTs:   0xFFFFFFF0,
		Width:   1280,
		Height:  720,
		Length:  uint32(FrameLength(1280, 720)),
		Lost:    3,
		Errors:  2,
		Dropped: 0x01020304,
		Skipped: 1,
		Queued:  0x0506,
		Pending: 4,
	}
	var b [Size]byte
	for i := range b {
		b[i] = 0xEE // what Marshal leaves alone would show
	}
	h.Marshal(b[:])
	if string(b[0:4]) != "MSFR" || b[4] != Version || b[7] != 0 {
		t.Fatalf("header bytes %x", b[:8])
	}
	got, err := Unmarshal(b[:])
	if err != nil {
		t.Fatal(err)
	}
	if got != h {
		t.Fatalf("got %+v, want %+v", got, h)
	}
	// Little-endian, as the producer's struct.unpack("<4sBBBBIqIHHIIIIIHH4x")
	// reads it.
	if b[8] != 41 || b[24] != 0x00 || b[25] != 0x05 {
		t.Fatalf("byte order: %x", b[8:28])
	}
	if b[40] != 0x04 || b[43] != 0x01 || b[48] != 0x06 || b[49] != 0x05 {
		t.Fatalf("byte order of the counts: %x", b[32:52])
	}
	if b[52] != 0 || b[53] != 0 || b[54] != 0 || b[55] != 0 {
		t.Fatalf("reserved bytes %x", b[52:56])
	}
}

func TestUnmarshalRefusesOtherBytes(t *testing.T) {
	var b [Size]byte
	Header{Width: 4, Height: 2, Length: 12}.Marshal(b[:])
	cases := map[string]func([]byte){
		"short":   func(b []byte) {},
		"magic":   func(b []byte) { b[0] = 'X' },
		"version": func(b []byte) { b[4] = 1 },
		"length":  func(b []byte) { b[28] = 13 },
	}
	for name, spoil := range cases {
		c := b
		spoil(c[:])
		in := c[:]
		if name == "short" {
			in = c[:Size-1]
		}
		if _, err := Unmarshal(in); !errors.Is(err, ErrHeader) {
			t.Fatalf("%s: %v", name, err)
		}
	}
}

func TestCountsStopAtTheirMaximum(t *testing.T) {
	if Count32(7) != 7 || Count32(math.MaxUint32+5) != math.MaxUint32 {
		t.Fatal("Count32")
	}
	if Count16(7) != 7 || Count16(-1) != 0 || Count16(math.MaxUint16+5) != math.MaxUint16 {
		t.Fatal("Count16")
	}
}
