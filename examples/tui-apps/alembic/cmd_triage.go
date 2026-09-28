package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"alembic/harness"
	"alembic/jev"
)

// alembic triage: the cron form of the scheduler. Same scoring, same log,
// same outbox rule as the TUI, no terminal.

type triageCLI struct {
	feed, outbox, log string
	once, jsonOut     bool
	interval          time.Duration
	tasks, calls      int
	live, demo        bool
}

func runTriageCLI(args []string, stdout, stderr io.Writer) int {
	base, _ := defaultConfig()
	var o triageCLI
	fs := flag.NewFlagSet("alembic triage", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&o.feed, "feed", base.Feed, "harness feed (JSONL) [$ALEMBIC_FEED]")
	fs.StringVar(&o.outbox, "outbox", base.Outbox, "command outbox (JSONL) [$ALEMBIC_OUTBOX]")
	fs.StringVar(&o.log, "log", filepath.Join(filepath.Dir(base.Receipts), "triage.jsonl"), "append-only triage log")
	fs.BoolVar(&o.once, "once", false, "run one tick and exit")
	fs.BoolVar(&o.jsonOut, "json", false, "print the Triage as JSON instead of a table")
	fs.DurationVar(&o.interval, "interval", 60*time.Second, "tick interval when looping")
	fs.IntVar(&o.tasks, "tasks", 8, "tasks sent to Jev per tick (0 = deterministic only)")
	fs.IntVar(&o.calls, "calls", 60, "Jev calls per rolling hour")
	fs.BoolVar(&o.live, "live", false, "call the real Jev API (TYPESAFE_API_KEY); mock otherwise")
	fs.BoolVar(&o.demo, "demo", false, "seed and use the demo feed under ~/.alembic/demo")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: alembic triage [--feed PATH] [--outbox PATH] [--once] [--json] [--interval 60s] [--tasks 8] [--calls 60] [--live] [--demo]")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	if o.demo {
		cfg := base
		cfg.Demo = true
		if err := setupDemo(&cfg); err != nil {
			fmt.Fprintln(stderr, "alembic triage: demo:", err)
			return 1
		}
		o.feed, o.outbox, o.log = cfg.Feed, cfg.Outbox, cfg.TriageLog
	}
	if o.interval < time.Second {
		o.interval = time.Second
	}
	client := jev.NewFromEnv()
	if !o.live {
		client.APIKey = "" // never spend a key unless asked to
	}
	budget := jev.DefaultBudget()
	budget.MaxTasksPerTick, budget.MaxCallsPerHour = o.tasks, o.calls
	tr := jev.NewTriager(client, budget)
	outbox := harness.NewOutbox(o.outbox)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// the previous tick comes from the log so --once under cron still sees change
	var prev *jev.Triage
	if last := jev.LoadTriage(o.log, 1); len(last) == 1 {
		prev = &last[0]
	}
	for {
		snap, _, err := harness.LoadAll(o.feed)
		if err != nil {
			fmt.Fprintln(stderr, "alembic triage: feed:", err)
			return 1
		}
		t := tr.Run(ctx, snap, pendingPingsFromOutbox(o.outbox, snap))
		if err := jev.AppendTriage(o.log, t); err != nil {
			fmt.Fprintln(stderr, "alembic triage: log:", err)
		}
		if triageChanged(prev, t) {
			if _, err := outbox.Send(harness.Command{Type: "triage", Data: t}); err != nil {
				fmt.Fprintln(stderr, "alembic triage: outbox:", err)
			}
		}
		cur := t
		prev = &cur
		if o.jsonOut {
			enc := json.NewEncoder(stdout)
			enc.SetIndent("", "  ")
			_ = enc.Encode(t)
		} else {
			printTriageTable(stdout, t, snap)
		}
		if o.once {
			return 0
		}
		select {
		case <-ctx.Done():
			return 0
		case <-time.After(o.interval):
		}
	}
}

// pendingPingsFromOutbox counts pings in the outbox per task that have no
// ping.ack in the feed yet: the cron form has no session to remember them.
func pendingPingsFromOutbox(path string, snap *harness.Snapshot) map[string]int {
	fh, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer fh.Close()
	acked := map[string]bool{}
	for _, a := range snap.Acks {
		acked[a.PingID] = true
	}
	out := map[string]int{}
	sc := bufio.NewScanner(fh)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for sc.Scan() {
		var c harness.Command
		if json.Unmarshal(sc.Bytes(), &c) != nil || c.Type != "ping" || c.TaskID == "" || acked[c.ID] {
			continue
		}
		out[c.TaskID]++
	}
	return out
}

func printTriageTable(w io.Writer, t jev.Triage, snap *harness.Snapshot) {
	mode := "jev"
	if t.Mock {
		mode = "jev-mock"
	}
	if t.Deterministic {
		mode = "deterministic"
		if t.Skipped != "" {
			mode += " (" + t.Skipped + ")"
		}
	}
	fmt.Fprintf(w, "triage %s · %d tasks · %s · %d calls · $%.4f · %s\n", t.At.Local().Format("15:04:05"), len(t.Tasks), mode, t.JevCalls, t.CostUSD, t.Latency.Round(time.Millisecond))
	fmt.Fprintf(w, "%-4s %-5s %-6s %-3s %-8s %s\n", "rank", "score", "next", "jev", "id", "title · reasons")
	for _, tt := range t.Tasks {
		title := ""
		if task, ok := snap.Tasks[tt.ID]; ok {
			title = oneLine(task.Title)
		}
		jv := "-"
		if tt.JevUsed {
			jv = "◆"
		}
		fmt.Fprintf(w, "%-4d %-5.2f %-6s %-3s %-8s %s · %s\n", tt.Rank, tt.Score, tt.Next, jv, tt.ID, title, strings.Join(tt.Reasons, " · "))
	}
}
