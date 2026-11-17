//go:build linux

package benchmark

import (
	"errors"
	"os"
	"syscall"
)

func openRegular(path string, flags int) (*os.File, error) {
	f, err := os.OpenFile(path, flags|syscall.O_NOFOLLOW|syscall.O_NONBLOCK, 0600)
	if err != nil {
		return nil, errors.New("file_open_refused")
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, errors.New("file_stat_failed")
	}
	raw, ok := st.Sys().(*syscall.Stat_t)
	if !st.Mode().IsRegular() || !ok || raw.Nlink != 1 {
		f.Close()
		return nil, errors.New("regular_single_link_file_required")
	}
	return f, nil
}
func lockFile(f *os.File) error {
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		return errors.New("journal_busy")
	}
	return nil
}
