package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	"github.com/relux-works/weightspace-index/benchmark"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	code := benchmark.Main(ctx, os.Args[1:], os.Stdout, os.Stderr)
	stop()
	os.Exit(code)
}
