package record

import (
	"errors"
	"testing"
)

func TestHeaderRoundTrip(t *testing.T) {
	h := Header{
		Flags:  FlagNTPValid | FlagKeyframe,
		Codec:  CodecH264,
		Seq:    41,
		NTPNs:  1_757_500_000_123_456_789,
		RTPTs:  0xFFFFFFF0,
		Width:  1280,
		Height: 720,
		Length: uint32(FrameLength(1280, 720)),
	}
	var b [Size]byte
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
	// Little-endian, as the producer's struct.unpack("<4sBBBBIqIHHI") reads it.
	if b[8] != 41 || b[24] != 0x00 || b[25] != 0x05 {
		t.Fatalf("byte order: %x", b[8:28])
	}
}

func TestUnmarshalRefusesOtherBytes(t *testing.T) {
	var b [Size]byte
	Header{Width: 4, Height: 2, Length: 12}.Marshal(b[:])
	cases := map[string]func([]byte){
		"short":   func(b []byte) {},
		"magic":   func(b []byte) { b[0] = 'X' },
		"version": func(b []byte) { b[4] = 2 },
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
