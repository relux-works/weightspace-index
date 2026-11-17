//go:build !linux

package benchmark

import (
	"errors"
	"os"
)

func openRegular(string, int) (*os.File, error) { return nil, errors.New("linux_required") }
func lockFile(*os.File) error                   { return errors.New("linux_required") }
