package harness

import (
	"bufio"
	"encoding/json"
	"errors"
	"io"
	"os"
	"time"
)

// Feed tails an append-only JSONL file. Each Poll returns the records
// appended since the previous call; a truncated file restarts from zero.
type Feed struct {
	Path   string
	offset int64
	buf    []byte
}

func NewFeed(path string) *Feed { return &Feed{Path: path} }

// Poll reads new records. A missing file is not an error: the harness may
// not have written anything yet.
func (f *Feed) Poll() ([]FeedRecord, error) {
	fh, err := os.Open(f.Path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	defer fh.Close()
	st, err := fh.Stat()
	if err != nil {
		return nil, err
	}
	if st.Size() < f.offset {
		f.offset, f.buf = 0, nil
	}
	if st.Size() == f.offset {
		return nil, nil
	}
	if _, err := fh.Seek(f.offset, io.SeekStart); err != nil {
		return nil, err
	}
	data, err := io.ReadAll(fh)
	if err != nil {
		return nil, err
	}
	f.offset += int64(len(data))
	f.buf = append(f.buf, data...)

	// only complete lines are parsed; an unterminated trailing partial line
	// stays in the buffer for the next poll so it is delivered exactly once
	var complete, rest []byte
	if last := lastNewline(f.buf); last < 0 {
		rest = f.buf
	} else {
		complete, rest = f.buf[:last+1], f.buf[last+1:]
	}
	var out []FeedRecord
	sc := bufio.NewScanner(bytesReader(complete))
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var r FeedRecord
		if err := json.Unmarshal(line, &r); err != nil {
			continue // one bad line never poisons the feed
		}
		if r.TS.IsZero() {
			r.TS = time.Now()
		}
		out = append(out, r)
	}
	f.buf = append([]byte{}, rest...)
	return out, nil
}

func lastNewline(b []byte) int {
	for i := len(b) - 1; i >= 0; i-- {
		if b[i] == '\n' {
			return i
		}
	}
	return -1
}

type byteReader struct {
	b []byte
	i int
}

func bytesReader(b []byte) io.Reader { return &byteReader{b: b} }

func (r *byteReader) Read(p []byte) (int, error) {
	if r.i >= len(r.b) {
		return 0, io.EOF
	}
	n := copy(p, r.b[r.i:])
	r.i += n
	return n, nil
}

// LoadAll folds an entire feed file into a fresh snapshot (startup).
func LoadAll(path string) (*Snapshot, *Feed, error) {
	f := NewFeed(path)
	s := NewSnapshot()
	recs, err := f.Poll()
	if err != nil {
		return s, f, err
	}
	for _, r := range recs {
		s.Apply(r)
	}
	return s, f, nil
}
