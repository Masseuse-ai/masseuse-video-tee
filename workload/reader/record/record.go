// Package record is the frame record stream-reader hands the producer: a
// fixed 32-byte header, then one I420 (yuv420p) frame of exactly
// Width x Height, so the reader on the other side of the pipe reads a
// header, then Length bytes, and never has to find its place.
//
// The header carries what the decode threw away: the time the frame
// belongs to on the sender's own clock (the RTSP sender report's NTP time
// for its RTP timestamp, interpolated to this frame), whether that time is
// known yet, and the RTP timestamp itself, which paces frames when it is
// not. Byte order is little-endian throughout.
//
//	offset  size  field
//	0       4     magic "MSFR"
//	4       1     version (1)
//	5       1     flags: bit 0 NTP valid, bit 1 keyframe
//	6       1     codec: 1 H.264, 2 H.265
//	7       1     reserved (0)
//	8       4     seq      access unit sequence, from 0, per connection
//	12      8     ntp_ns   unix nanoseconds; 0 when the NTP flag is clear
//	20      4     rtp_ts   the access unit's RTP timestamp (90 kHz)
//	24      2     width
//	26      2     height
//	28      4     length   width*height*3/2, the frame bytes that follow
//
// producer/producer.py (Decoder) unpacks the same layout.
package record

import (
	"encoding/binary"
	"errors"
	"fmt"
)

// Size is the header's size in bytes.
const Size = 32

// Version is the layout this package writes.
const Version = 1

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
}

// FrameLength is the byte count of a Width x Height I420 frame.
func FrameLength(width, height int) int {
	return width * height * 3 / 2
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
		Flags:  b[5],
		Codec:  b[6],
		Seq:    binary.LittleEndian.Uint32(b[8:12]),
		NTPNs:  int64(binary.LittleEndian.Uint64(b[12:20])),
		RTPTs:  binary.LittleEndian.Uint32(b[20:24]),
		Width:  binary.LittleEndian.Uint16(b[24:26]),
		Height: binary.LittleEndian.Uint16(b[26:28]),
		Length: binary.LittleEndian.Uint32(b[28:32]),
	}
	if want := FrameLength(int(h.Width), int(h.Height)); int(h.Length) != want {
		return Header{}, fmt.Errorf("%w: length %d for %dx%d", ErrHeader, h.Length, h.Width, h.Height)
	}
	return h, nil
}
