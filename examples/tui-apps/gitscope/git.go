package main

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
)

const fieldSep = "\x1f"

type statusEntry struct {
	Path     string
	OrigPath string
	Code     string
	Staged   bool
}

type repoStatus struct {
	Head     string
	OID      string
	Upstream string
	Ahead    int
	Behind   int
	Detached bool
	Unborn   bool
	Entries  []statusEntry
	Raw      string
}

type branch struct {
	Name     string
	Current  bool
	Upstream string
	Short    string
	Subject  string
}

type commit struct {
	SHA     string
	Short   string
	Author  string
	Age     string
	Subject string
}

type stash struct {
	Ref     string
	Message string
}

type statusMsg struct {
	st  repoStatus
	err error
}

type branchesMsg struct {
	branches []branch
	err      error
}

type commitsMsg struct {
	commits []commit
	err     error
}

type stashesMsg struct {
	stashes []stash
	err     error
}

type diffReq struct {
	key   string
	title string
	kind  string
	path  string
	args  []string
}

type diffMsg struct {
	seq   int
	key   string
	title string
	kind  string
	path  string
	text  string
	err   error
}

type actionMsg struct {
	verb string
	ok   string
	out  string
	err  error
}

type copiedMsg struct {
	sha string
	err error
}

func runGit(root string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	cmd.Env = append(os.Environ(), "GIT_OPTIONAL_LOCKS=0", "LC_ALL=C")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = err.Error()
		}
		return stdout.String(), errors.New(firstLine(msg))
	}
	return stdout.String(), nil
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

func repoRoot(dir string) (string, error) {
	out, err := runGit(dir, "rev-parse", "--show-toplevel")
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(out), nil
}

func parseStatus(out string) repoStatus {
	st := repoStatus{Raw: out}
	var staged, unstaged, untracked []statusEntry
	for _, line := range strings.Split(out, "\n") {
		if line == "" {
			continue
		}
		switch line[0] {
		case '#':
			parseStatusHeader(&st, line)
		case '1':
			parts := strings.SplitN(line, " ", 9)
			if len(parts) < 9 {
				continue
			}
			addXY(parts[1], parts[8], "", &staged, &unstaged)
		case '2':
			parts := strings.SplitN(line, " ", 10)
			if len(parts) < 10 {
				continue
			}
			path, orig, _ := strings.Cut(parts[9], "\t")
			addXY(parts[1], path, orig, &staged, &unstaged)
		case 'u':
			parts := strings.SplitN(line, " ", 11)
			if len(parts) < 11 {
				continue
			}
			unstaged = append(unstaged, statusEntry{Path: parts[10], Code: "U"})
		case '?':
			untracked = append(untracked, statusEntry{Path: line[2:], Code: "?"})
		}
	}
	st.Entries = append(append(staged, unstaged...), untracked...)
	return st
}

func parseStatusHeader(st *repoStatus, line string) {
	fields := strings.Fields(line)
	if len(fields) < 3 {
		return
	}
	switch fields[1] {
	case "branch.oid":
		st.OID = fields[2]
		st.Unborn = fields[2] == "(initial)"
	case "branch.head":
		st.Head = fields[2]
		st.Detached = fields[2] == "(detached)"
	case "branch.upstream":
		st.Upstream = fields[2]
	case "branch.ab":
		if len(fields) >= 4 {
			st.Ahead, _ = strconv.Atoi(strings.TrimPrefix(fields[2], "+"))
			st.Behind, _ = strconv.Atoi(strings.TrimPrefix(fields[3], "-"))
		}
	}
}

func addXY(xy, path, orig string, staged, unstaged *[]statusEntry) {
	if len(xy) != 2 {
		return
	}
	if xy[0] != '.' {
		*staged = append(*staged, statusEntry{Path: path, OrigPath: orig, Code: string(xy[0]), Staged: true})
	}
	if xy[1] != '.' {
		*unstaged = append(*unstaged, statusEntry{Path: path, Code: string(xy[1])})
	}
}

func parseBranches(out string) []branch {
	var bs []branch
	for _, line := range strings.Split(out, "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		f := strings.Split(line, fieldSep)
		if len(f) < 5 {
			continue
		}
		bs = append(bs, branch{
			Current:  strings.TrimSpace(f[0]) == "*",
			Name:     f[1],
			Upstream: f[2],
			Short:    f[3],
			Subject:  f[4],
		})
	}
	return bs
}

func parseCommits(out string) []commit {
	var cs []commit
	for _, line := range strings.Split(out, "\n") {
		if line == "" {
			continue
		}
		f := strings.Split(line, fieldSep)
		if len(f) < 5 {
			continue
		}
		cs = append(cs, commit{SHA: f[0], Short: f[1], Author: f[2], Age: shortAge(f[3]), Subject: f[4]})
	}
	return cs
}

func parseStashes(out string) []stash {
	var ss []stash
	for _, line := range strings.Split(out, "\n") {
		if line == "" {
			continue
		}
		ref, msg, _ := strings.Cut(line, fieldSep)
		ss = append(ss, stash{Ref: ref, Message: msg})
	}
	return ss
}

func shortAge(rel string) string {
	rel = strings.TrimSuffix(rel, " ago")
	f := strings.Fields(rel)
	if len(f) < 2 {
		return rel
	}
	unit := strings.TrimSuffix(f[1], "s")
	abbr := map[string]string{
		"second": "s", "minute": "m", "hour": "h", "day": "d",
		"week": "w", "month": "mo", "year": "y",
	}
	if a, ok := abbr[unit]; ok {
		return f[0] + a
	}
	return rel
}

func loadStatus(root string) tea.Cmd {
	return func() tea.Msg {
		out, err := runGit(root, "status", "--porcelain=v2", "--branch", "--untracked-files=normal")
		if err != nil {
			return statusMsg{err: err}
		}
		return statusMsg{st: parseStatus(out)}
	}
}

func loadBranches(root string) tea.Cmd {
	return func() tea.Msg {
		format := "%(HEAD)" + fieldSep + "%(refname:short)" + fieldSep + "%(upstream:short)" + fieldSep + "%(objectname:short)" + fieldSep + "%(contents:subject)"
		out, err := runGit(root, "branch", "--format="+format)
		if err != nil {
			return branchesMsg{err: err}
		}
		return branchesMsg{branches: parseBranches(out)}
	}
}

func loadCommits(root string) tea.Cmd {
	return func() tea.Msg {
		out, err := runGit(root, "log", "--no-color", "-n", "300", "--format=%H"+fieldSep+"%h"+fieldSep+"%an"+fieldSep+"%ar"+fieldSep+"%s")
		if err != nil {
			if strings.Contains(err.Error(), "does not have any commits") || strings.Contains(err.Error(), "unknown revision") {
				return commitsMsg{}
			}
			return commitsMsg{err: err}
		}
		return commitsMsg{commits: parseCommits(out)}
	}
}

func loadStashes(root string) tea.Cmd {
	return func() tea.Msg {
		out, err := runGit(root, "stash", "list", "--format=%gd"+fieldSep+"%s")
		if err != nil {
			return stashesMsg{err: err}
		}
		return stashesMsg{stashes: parseStashes(out)}
	}
}

const maxFileBytes = 512 * 1024

func loadDiff(root string, seq int, req diffReq) tea.Cmd {
	return func() tea.Msg {
		msg := diffMsg{seq: seq, key: req.key, title: req.title, kind: req.kind, path: req.path}
		if req.kind == "file" {
			msg.text, msg.err = readUntracked(root, req.path)
			return msg
		}
		out, err := runGit(root, req.args...)
		msg.text, msg.err = out, err
		return msg
	}
}

func readUntracked(root, path string) (string, error) {
	full := filepath.Join(root, path)
	info, err := os.Stat(full)
	if err != nil {
		return "", err
	}
	if info.IsDir() {
		out, err := runGit(root, "ls-files", "--others", "--exclude-standard", "--", path)
		if err != nil {
			return "", err
		}
		return out, nil
	}
	f, err := os.Open(full)
	if err != nil {
		return "", err
	}
	defer f.Close()
	buf := make([]byte, maxFileBytes)
	n, _ := f.Read(buf)
	buf = buf[:n]
	if bytes.IndexByte(buf, 0) >= 0 {
		return fmt.Sprintf("binary file (%d bytes)", info.Size()), nil
	}
	text := string(buf)
	if info.Size() > int64(n) {
		text += fmt.Sprintf("\n… truncated (%d bytes total)", info.Size())
	}
	return text, nil
}

func gitAction(root, verb, ok string, argSets ...[]string) tea.Cmd {
	return func() tea.Msg {
		var out string
		for _, args := range argSets {
			o, err := runGit(root, args...)
			if err != nil {
				return actionMsg{verb: verb, err: err}
			}
			out = o
		}
		return actionMsg{verb: verb, ok: ok, out: strings.TrimSpace(out)}
	}
}

func removePath(root, path, ok string) tea.Cmd {
	return func() tea.Msg {
		full := filepath.Join(root, path)
		if err := os.RemoveAll(full); err != nil {
			return actionMsg{verb: "discard", err: err}
		}
		return actionMsg{verb: "discard", ok: ok}
	}
}

func stageToggleArgs(e statusEntry) []string {
	if e.Staged {
		return []string{"restore", "--staged", "--", e.Path}
	}
	return []string{"add", "--", e.Path}
}

func discardArgSets(e statusEntry) [][]string {
	if e.Staged {
		if e.Code == "A" {
			return [][]string{{"restore", "--staged", "--", e.Path}}
		}
		return [][]string{{"restore", "--staged", "--", e.Path}, {"checkout", "--", e.Path}}
	}
	return [][]string{{"checkout", "--", e.Path}}
}

func diffRequest(e statusEntry) diffReq {
	switch {
	case e.Code == "?":
		return diffReq{key: "untracked:" + e.Path, title: "Untracked · " + e.Path, kind: "file", path: e.Path}
	case e.Staged:
		args := []string{"diff", "--cached", "--no-color", "--"}
		if e.OrigPath != "" {
			args = append(args, e.OrigPath)
		}
		args = append(args, e.Path)
		return diffReq{key: "staged:" + e.Path, title: "Staged · " + e.Path, kind: "diff", path: e.Path, args: args}
	default:
		return diffReq{key: "unstaged:" + e.Path, title: "Unstaged · " + e.Path, kind: "diff", path: e.Path,
			args: []string{"diff", "--no-color", "--", e.Path}}
	}
}

func commitRequest(c commit) diffReq {
	return diffReq{key: "commit:" + c.SHA, title: "Commit · " + c.Short, kind: "diff",
		args: []string{"show", "--no-color", "--stat", "-p", c.SHA}}
}

func branchRequest(b branch) diffReq {
	return diffReq{key: "branch:" + b.Name, title: "Branch · " + b.Name + " · last 15 commits", kind: "log",
		args: []string{"log", "--no-color", "-n", "15", "--format=%h" + fieldSep + "%ar" + fieldSep + "%an" + fieldSep + "%s", b.Name, "--"}}
}

func stashRequest(s stash) diffReq {
	return diffReq{key: "stash:" + s.Ref, title: "Stash · " + s.Ref, kind: "diff",
		args: []string{"stash", "show", "-p", "--stat", "--no-color", s.Ref}}
}
