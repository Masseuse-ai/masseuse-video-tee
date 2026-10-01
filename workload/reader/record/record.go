// Package record is the frame record stream-reader hands the producer: a
// fixed 56-byte header, then one I420 (yuv420p) frame of exactly
// Width x Height, so the reader on the other side of the pipe reads a
// header, then Length bytes, and never has to find its place.
//
// The header carries what the decode threw away: the time the frame
// belongs to on the sender's own clock (the RTSP sender report's NTP time
// for its RTP timestamp, interpolated to this frame), whether that time is
// known yet, and the RTP timestamp itself, which paces frames when it is
// not. It also carries what went wrong on the way since the connection's
// first keyframe, and how much was waiting for the decoder as the frame
// came out. Byte order is little-endian throughout.
//
//	offset  size  field
//	0       4     magic "MSFR"
//	4       1     version (2)
//	5       1     flags: bit 0 NTP valid, bit 1 keyframe
//	6       1     codec: 1 H.264, 2 H.265
//	7       1     reserved (0)
//	8       4     seq      access unit sequence, from 0, per connection
//	12      8     ntp_ns   unix nanoseconds; 0 when the NTP flag is clear
//	20      4     rtp_ts   the access unit's RTP timestamp (90 kHz)
//	24      2     width
//	26      2     height
//	28      4     length   width*height*3/2, the frame bytes that follow
//	32      4     lost     RTP packets that never arrived
//	36      4     errors   packets that could not be read or depacketized
//	40      4     dropped  access units left out: damaged, or the decoder behind
//	44      4     skipped  access units the decoder gave no frame for
//	48      2     queued   access units waiting for the decoder
//	50      2     pending  access units inside the decoder
//	52      4     reserved (0)
//
// lost, errors, dropped and skipped are running counts for the connection
// and stop at their maximum rather than wrap; queued and pending are as the
// frame was written. seq counts the units the stream gave, so a gap in it
// is the units left out or skipped between two records.
//
// producer/producer.py (Decoder) unpacks the same layout.
package record

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// Size is the header's size in bytes.
const Size = 56

// Version is the layout this package writes.
const Version = 2

// Flags.
const (
	FlagNTPValid = 1 << 0
	FlagKeyframe = 1 << 1
)

// Codecs.
const (
	CodecH264 uint8 = 1
	CodecH265 uint8 = 2
)

var magic = [4]byte{'M', 'S', 'F', 'R'}

// ErrHeader is returned by Unmarshal for bytes that are not a header this
// package wrote.
var ErrHeader = errors.New("record: not a frame header")

// Header is one record's header.
type Header struct {
	Flags  uint8
	Codec  uint8
	Seq    uint32
	NTPNs  int64
	RTPTs  uint32
	Width  uint16
	Height uint16
	Length uint32

	Lost    uint32
	Errors  uint32
	Dropped uint32
	Skipped uint32
	Queued  uint16
	Pending uint16
}

// FrameLength is the byte count of a Width x Height I420 frame.
func FrameLength(width, height int) int {
	return width * height * 3 / 2
}

// Count32 is a running count as the header carries it.
func Count32(n uint64) uint32 {
	return uint32(min(n, math.MaxUint32))
}

// Count16 is a depth as the header carries it.
func Count16(n int) uint16 {
	return uint16(max(0, min(n, math.MaxUint16)))
}

// Marshal writes the header into dst, which must hold Size bytes.
func (h Header) Marshal(dst []byte) {
	copy(dst[0:4], magic[:])
	dst[4] = Version
	dst[5] = h.Flags
	dst[6] = h.Codec
	dst[7] = 0
	binary.LittleEndian.PutUint32(dst[8:12], h.Seq)
	binary.LittleEndian.PutUint64(dst[12:20], uint64(h.NTPNs))
	binary.LittleEndian.PutUint32(dst[20:24], h.RTPTs)
	binary.LittleEndian.PutUint16(dst[24:26], h.Width)
	binary.LittleEndian.PutUint16(dst[26:28], h.Height)
	binary.LittleEndian.PutUint32(dst[28:32], h.Length)
	binary.LittleEndian.PutUint32(dst[32:36], h.Lost)
	binary.LittleEndian.PutUint32(dst[36:40], h.Errors)
	binary.LittleEndian.PutUint32(dst[40:44], h.Dropped)
	binary.LittleEndian.PutUint32(dst[44:48], h.Skipped)
	binary.LittleEndian.PutUint16(dst[48:50], h.Queued)
	binary.LittleEndian.PutUint16(dst[50:52], h.Pending)
	clear(dst[52:56])
}

// Unmarshal reads a header from the first Size bytes of b.
func Unmarshal(b []byte) (Header, error) {
	if len(b) < Size {
		return Header{}, fmt.Errorf("%w: %d bytes", ErrHeader, len(b))
	}
	if [4]byte{b[0], b[1], b[2], b[3]} != magic {
		return Header{}, fmt.Errorf("%w: magic %q", ErrHeader, b[0:4])
	}
	if b[4] != Version {
		return Header{}, fmt.Errorf("%w: version %d", ErrHeader, b[4])
	}
	h := Header{
		Flags:   b[5],
		Codec:   b[6],
		Seq:     binary.LittleEndian.Uint32(b[8:12]),
		NTPNs:   int64(binary.LittleEndian.Uint64(b[12:20])),
		RTPTs:   binary.LittleEndian.Uint32(b[20:24]),
		Width:   binary.LittleEndian.Uint16(b[24:26]),
		Height:  binary.LittleEndian.Uint16(b[26:28]),
		Length:  binary.LittleEndian.Uint32(b[28:32]),
		Lost:    binary.LittleEndian.Uint32(b[32:36]),
		Errors:  binary.LittleEndian.Uint32(b[36:40]),
		Dropped: binary.LittleEndian.Uint32(b[40:44]),
		Skipped: binary.LittleEndian.Uint32(b[44:48]),
		Queued:  binary.LittleEndian.Uint16(b[48:50]),
		Pending: binary.LittleEndian.Uint16(b[50:52]),
	}
	if want := FrameLength(int(h.Width), int(h.Height)); int(h.Length) != want {
		return Header{}, fmt.Errorf("%w: length %d for %dx%d", ErrHeader, h.Length, h.Width, h.Height)
	}
	return h, nil
}
