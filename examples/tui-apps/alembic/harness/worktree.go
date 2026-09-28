package harness

import (
	"bytes"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// Worktree is one entry of `git worktree list --porcelain`, enriched.
type Worktree struct {
	Path     string
	Branch   string // "" when detached
	Head     string
	Detached bool
	Main     bool
	Dirty    int // changed files in the working tree
	Ahead    int
	Behind   int
	Upstream string
	TaskIDs  []string // tasks whose Worktree matches Path
}

func git(dir string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(errb.String())
		if msg == "" {
			msg = err.Error()
		}
		return "", &GitError{Args: args, Msg: msg}
	}
	return out.String(), nil
}

type GitError struct {
	Args []string
	Msg  string
}

func (e *GitError) Error() string { return "git " + strings.Join(e.Args, " ") + ": " + e.Msg }

// RepoRoot returns the top-level directory of the repo containing dir.
func RepoRoot(dir string) (string, error) {
	out, err := git(dir, "rev-parse", "--show-toplevel")
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(out), nil
}

// ListWorktrees parses `git worktree list --porcelain` and enriches each entry.
func ListWorktrees(repo string) ([]Worktree, error) {
	out, err := git(repo, "worktree", "list", "--porcelain")
	if err != nil {
		return nil, err
	}
	var wts []Worktree
	var cur *Worktree
	for _, line := range strings.Split(out, "\n") {
		switch {
		case strings.HasPrefix(line, "worktree "):
			wts = append(wts, Worktree{Path: strings.TrimPrefix(line, "worktree ")})
			cur = &wts[len(wts)-1]
		case cur == nil:
		case strings.HasPrefix(line, "HEAD "):
			cur.Head = strings.TrimPrefix(line, "HEAD ")
		case strings.HasPrefix(line, "branch "):
			cur.Branch = strings.TrimPrefix(strings.TrimPrefix(line, "branch "), "refs/heads/")
		case line == "detached":
			cur.Detached = true
		}
	}
	for i := range wts {
		wts[i].Main = i == 0
		enrich(&wts[i])
	}
	return wts, nil
}

func enrich(w *Worktree) {
	if out, err := git(w.Path, "status", "--porcelain"); err == nil {
		n := 0
		for _, l := range strings.Split(out, "\n") {
			if strings.TrimSpace(l) != "" {
				n++
			}
		}
		w.Dirty = n
	}
	if w.Branch == "" {
		return
	}
	if up, err := git(w.Path, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"); err == nil {
		w.Upstream = strings.TrimSpace(up)
		if out, err := git(w.Path, "rev-list", "--left-right", "--count", "HEAD...@{upstream}"); err == nil {
			f := strings.Fields(out)
			if len(f) == 2 {
				w.Ahead, _ = strconv.Atoi(f[0])
				w.Behind, _ = strconv.Atoi(f[1])
			}
		}
	}
}

// AddWorktree creates ../<repo>-<branch> (or path if given) on a new branch.
func AddWorktree(repo, branch, path string) (string, error) {
	if _, err := git(repo, "check-ref-format", "--branch", branch); err != nil {
		return "", &GitError{Args: []string{"worktree", "add"}, Msg: "invalid branch name: " + branch}
	}
	if path == "" {
		base := filepath.Base(repo)
		safe := strings.NewReplacer("/", "-", " ", "-").Replace(branch)
		path = filepath.Join(filepath.Dir(repo), base+"-"+safe)
	}
	if _, err := git(repo, "worktree", "add", "-b", branch, path); err != nil {
		// branch may already exist: fall back to checking it out
		if _, err2 := git(repo, "worktree", "add", path, branch); err2 != nil {
			return "", err
		}
	}
	return path, nil
}

func RemoveWorktree(repo, path string, force bool) error {
	args := []string{"worktree", "remove"}
	if force {
		args = append(args, "--force")
	}
	_, err := git(repo, append(args, path)...)
	return err
}

// LinkTasks fills TaskIDs from the snapshot by matching Worktree paths.
func LinkTasks(wts []Worktree, s *Snapshot) {
	for i := range wts {
		wts[i].TaskIDs = nil
		for _, t := range s.Tasks {
			if t.Worktree != "" && filepath.Clean(t.Worktree) == filepath.Clean(wts[i].Path) {
				wts[i].TaskIDs = append(wts[i].TaskIDs, t.ID)
			}
		}
	}
}

// StagedDiff returns the staged diff of a worktree (input for Jev packs).
func StagedDiff(dir string) (string, error) { return git(dir, "diff", "--cached", "--no-color") }

// WorkingDiff returns the unstaged diff of a worktree.
func WorkingDiff(dir string) (string, error) { return git(dir, "diff", "--no-color") }
