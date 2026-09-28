package main

import (
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/aymanbagabas/go-osc52/v2"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/harmonica"
	"github.com/charmbracelet/huh"
	"github.com/sahilm/fuzzy"
)

type paneID int

const (
	paneStatus paneID = iota
	paneBranches
	paneCommits
	paneStash
	paneDiff
)

const (
	leftPanes  = 4
	paneCount  = 5
	collapsedH = 3
	minWidth   = 60
	minHeight  = 14

	autoRefreshEvery = 3 * time.Second
	toastLife        = 2500 * time.Millisecond
	animFrame        = time.Second / 30
)

type modalKind int

const (
	modalNone modalKind = iota
	modalCommit
	modalBranch
	modalConfirm
)

type confirmKind int

const (
	confirmDiscard confirmKind = iota
	confirmStash
)

type formValues struct {
	title  string
	body   string
	branch string
	yes    bool
}

type listState struct {
	sel    int
	off    int
	filter string
}

type toast struct {
	id   int
	text string
	ok   bool
}

type clockMsg time.Time
type animMsg struct{}
type toastExpireMsg struct{ id int }

type model struct {
	root     string
	cwd      string
	repoName string
	noRepo   bool

	width  int
	height int

	focus    paneID
	lastLeft paneID
	lists    [leftPanes]listState

	status    repoStatus
	statusErr string
	branches  []branch
	commits   []commit
	stashes   []stash
	loaded    bool

	vp          viewport.Model
	diff        diffMsg
	diffKey     string
	diffSeq     int
	diffLoading bool

	spin     spinner.Model
	spinning bool
	busy     int

	toast   *toast
	toastID int

	showHelp      bool
	filterEditing bool

	modal        modalKind
	modalTitle   string
	form         *huh.Form
	fv           *formValues
	confirm      confirmKind
	confirmEntry statusEntry

	heights   [leftPanes]float64
	vels      [leftPanes]float64
	animating bool
	spring    harmonica.Spring

	now         time.Time
	lastRefresh time.Time
	lastGit     []string
}

func newModel(root, cwd string) model {
	m := model{
		root:     root,
		cwd:      cwd,
		repoName: filepath.Base(root),
		noRepo:   root == "",
		focus:    paneStatus,
		lastLeft: paneStatus,
		now:      time.Now(),
		spring:   harmonica.NewSpring(harmonica.FPS(30), 7.0, 1.0),
	}
	m.spin = spinner.New(spinner.WithSpinner(spinner.Dot), spinner.WithStyle(stGold))
	m.spinning = true
	m.vp = viewport.New(0, 0)
	m.vp.MouseWheelEnabled = false
	m.vp.KeyMap = viewport.KeyMap{}
	m.diff = diffMsg{kind: "empty", text: "Select a file, commit, branch or stash to inspect it."}
	return m
}

func (m model) Init() tea.Cmd {
	if m.noRepo {
		return nil
	}
	return tea.Batch(m.refreshAll(), clockTick(), m.spin.Tick)
}

func clockTick() tea.Cmd {
	return tea.Tick(time.Second, func(t time.Time) tea.Msg { return clockMsg(t) })
}

func animTick() tea.Cmd {
	return tea.Tick(animFrame, func(time.Time) tea.Msg { return animMsg{} })
}

func (m model) refreshAll() tea.Cmd {
	return tea.Batch(loadStatus(m.root), loadBranches(m.root), loadCommits(m.root), loadStashes(m.root))
}

func (m model) refreshLight() tea.Cmd {
	return tea.Batch(loadStatus(m.root), loadBranches(m.root))
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		return m.resize(msg.Width, msg.Height), nil
	case tea.KeyMsg:
		return m.handleKey(msg)
	case tea.MouseMsg:
		return m.handleMouse(msg)
	case clockMsg:
		m.now = time.Time(msg)
		var cmd tea.Cmd
		if !m.noRepo && m.modal == modalNone && time.Since(m.lastRefresh) >= autoRefreshEvery {
			m.lastRefresh = time.Now()
			cmd = m.refreshLight()
		}
		return m, tea.Batch(clockTick(), cmd)
	case animMsg:
		return m.stepAnimation()
	case spinner.TickMsg:
		if !m.diffLoading && m.busy == 0 && m.loaded {
			m.spinning = false
			return m, nil
		}
		var cmd tea.Cmd
		m.spin, cmd = m.spin.Update(msg)
		return m, cmd
	case toastExpireMsg:
		if m.toast != nil && m.toast.id == msg.id {
			m.toast = nil
		}
		return m, nil
	case statusMsg:
		return m.onStatus(msg)
	case branchesMsg:
		return m.onBranches(msg)
	case commitsMsg:
		return m.onCommits(msg)
	case stashesMsg:
		return m.onStashes(msg)
	case diffMsg:
		return m.onDiff(msg)
	case actionMsg:
		return m.onAction(msg)
	case copiedMsg:
		if msg.err != nil {
			return m.showToast("copy failed: "+msg.err.Error(), false)
		}
		return m.showToast("copied "+msg.sha, true)
	}
	if m.modal != modalNone && m.form != nil {
		return m.updateForm(msg)
	}
	return m, nil
}

// ---------------------------------------------------------------- layout

type layout struct {
	ok     bool
	leftW  int
	rightW int
	bodyH  int
	paneH  [leftPanes]int
	paneY  [leftPanes]int
}

func (m model) layout() layout {
	var l layout
	if m.width < minWidth || m.height < minHeight {
		return l
	}
	l.ok = true
	l.bodyH = m.height - 2
	l.leftW = maxInt(28, m.width*35/100)
	l.rightW = m.width - l.leftW
	sum := 0
	for i := range l.paneH {
		h := maxInt(collapsedH, int(math.Round(m.heights[i])))
		l.paneH[i] = h
		sum += h
	}
	l.paneH[m.lastLeft] = maxInt(collapsedH, l.paneH[m.lastLeft]+l.bodyH-sum)
	y := 1
	for i := range l.paneY {
		l.paneY[i] = y
		y += l.paneH[i]
	}
	return l
}

// hit maps a terminal cell to a pane and a content row inside it (-1 when the
// cell is on a border).
func (l layout) hit(x, y int) (paneID, int, bool) {
	if !l.ok || y < 1 || y >= 1+l.bodyH {
		return 0, -1, false
	}
	if x >= l.leftW {
		row := y - 2
		if row >= l.bodyH-2 {
			row = -1
		}
		return paneDiff, row, true
	}
	for i := 0; i < leftPanes; i++ {
		if y >= l.paneY[i] && y < l.paneY[i]+l.paneH[i] {
			row := y - l.paneY[i] - 1
			if row < 0 || row >= l.paneH[i]-2 {
				row = -1
			}
			return paneID(i), row, true
		}
	}
	return 0, -1, false
}

func (m model) targets() [leftPanes]float64 {
	var t [leftPanes]float64
	big := float64(m.height - 2 - collapsedH*(leftPanes-1))
	for i := range t {
		t[i] = collapsedH
	}
	t[m.lastLeft] = big
	return t
}

func (m model) snapHeights() model {
	m.heights = m.targets()
	m.vels = [leftPanes]float64{}
	m.animating = false
	return m
}

func (m model) stepAnimation() (tea.Model, tea.Cmd) {
	if !m.animating {
		return m, nil
	}
	t := m.targets()
	settled := true
	for i := range m.heights {
		m.heights[i], m.vels[i] = m.spring.Update(m.heights[i], m.vels[i], t[i])
		if math.Abs(m.heights[i]-t[i]) > 0.4 || math.Abs(m.vels[i]) > 0.4 {
			settled = false
		}
	}
	if settled {
		return m.snapHeights(), nil
	}
	return m, animTick()
}

func (m model) resize(w, h int) model {
	m.width, m.height = w, h
	m = m.snapHeights()
	if l := m.layout(); l.ok {
		m.vp.Width = l.rightW - 2
		m.vp.Height = l.bodyH - 2
		m = m.rerenderDiff()
	}
	return m
}

func (m model) rerenderDiff() model {
	y := m.vp.YOffset
	m.vp.SetContent(renderContent(m.diff, m.vp.Width))
	m.vp.SetYOffset(y)
	return m
}

// ---------------------------------------------------------------- lists

func filterIndices(pattern string, labels []string) []int {
	if pattern == "" {
		idx := make([]int, len(labels))
		for i := range labels {
			idx[i] = i
		}
		return idx
	}
	matches := fuzzy.Find(pattern, labels)
	idx := make([]int, len(matches))
	for i, mt := range matches {
		idx[i] = mt.Index
	}
	sort.Ints(idx)
	return idx
}

func (m model) visible(p paneID) []int {
	var labels []string
	switch p {
	case paneStatus:
		for _, e := range m.status.Entries {
			labels = append(labels, e.Path)
		}
	case paneBranches:
		for _, b := range m.branches {
			labels = append(labels, b.Name)
		}
	case paneCommits:
		for _, c := range m.commits {
			labels = append(labels, c.Short+" "+c.Subject+" "+c.Author)
		}
	case paneStash:
		for _, s := range m.stashes {
			labels = append(labels, s.Ref+" "+s.Message)
		}
	}
	return filterIndices(m.lists[p].filter, labels)
}

func (m model) clampList(p paneID) model {
	n := len(m.visible(p))
	m.lists[p].sel = clampInt(m.lists[p].sel, 0, maxInt(0, n-1))
	return m
}

func (m model) selectedEntry() (statusEntry, bool) {
	vis := m.visible(paneStatus)
	s := m.lists[paneStatus].sel
	if s < 0 || s >= len(vis) {
		return statusEntry{}, false
	}
	return m.status.Entries[vis[s]], true
}

func (m model) selectedBranch() (branch, bool) {
	vis := m.visible(paneBranches)
	s := m.lists[paneBranches].sel
	if s < 0 || s >= len(vis) {
		return branch{}, false
	}
	return m.branches[vis[s]], true
}

func (m model) selectedCommit() (commit, bool) {
	vis := m.visible(paneCommits)
	s := m.lists[paneCommits].sel
	if s < 0 || s >= len(vis) {
		return commit{}, false
	}
	return m.commits[vis[s]], true
}

func (m model) selectedStash() (stash, bool) {
	vis := m.visible(paneStash)
	s := m.lists[paneStash].sel
	if s < 0 || s >= len(vis) {
		return stash{}, false
	}
	return m.stashes[vis[s]], true
}

func (m model) currentRequest() (diffReq, bool) {
	switch m.lastLeft {
	case paneStatus:
		if e, ok := m.selectedEntry(); ok {
			return diffRequest(e), true
		}
	case paneBranches:
		if b, ok := m.selectedBranch(); ok {
			return branchRequest(b), true
		}
	case paneCommits:
		if c, ok := m.selectedCommit(); ok {
			return commitRequest(c), true
		}
	case paneStash:
		if s, ok := m.selectedStash(); ok {
			return stashRequest(s), true
		}
	}
	return diffReq{}, false
}

func (m model) emptyText() string {
	filtered := m.lists[m.lastLeft].filter != ""
	switch m.lastLeft {
	case paneStatus:
		if filtered {
			return "No files match the filter."
		}
		return "Working tree clean — nothing to stage."
	case paneBranches:
		if filtered {
			return "No branches match the filter."
		}
		return "No branches yet — make a first commit."
	case paneCommits:
		if filtered {
			return "No commits match the filter."
		}
		return "No commits yet."
	default:
		if filtered {
			return "No stashes match the filter."
		}
		return "Stash is empty."
	}
}

// selectChanged loads the right pane for the active list's selection.
func (m model) selectChanged(force, silent bool) (model, tea.Cmd) {
	req, ok := m.currentRequest()
	if !ok {
		m.diffSeq++
		m.diffLoading = false
		m.diffKey = fmt.Sprintf("empty:%d", m.lastLeft)
		m.diff = diffMsg{key: m.diffKey, kind: "empty", text: m.emptyText()}
		m.vp.GotoTop()
		return m.rerenderDiff(), nil
	}
	if req.key == m.diffKey && !force {
		return m, nil
	}
	m.diffSeq++
	m.diffKey = req.key
	cmds := []tea.Cmd{loadDiff(m.root, m.diffSeq, req)}
	if !silent {
		m.diffLoading = true
		m, c := m.spinStart()
		return m, tea.Batch(append(cmds, c)...)
	}
	return m, tea.Batch(cmds...)
}

func (m model) spinStart() (model, tea.Cmd) {
	if m.spinning {
		return m, nil
	}
	m.spinning = true
	return m, m.spin.Tick
}

func (m model) storeOffset(p paneID) model {
	l := m.layout()
	if !l.ok {
		return m
	}
	rows, cursor := m.rows(p, l.leftW-2)
	m.lists[p].off = viewOffset(m.lists[p].off, cursor, rows, l.paneH[p]-2)
	return m
}

func (m model) setSel(p paneID, i int) (model, tea.Cmd) {
	n := len(m.visible(p))
	if n == 0 {
		return m, nil
	}
	i = clampInt(i, 0, n-1)
	if i == m.lists[p].sel {
		return m, nil
	}
	m.lists[p].sel = i
	m = m.storeOffset(p)
	if p == m.lastLeft {
		return m.selectChanged(false, false)
	}
	return m, nil
}

func (m model) moveSel(p paneID, d int) (model, tea.Cmd) {
	return m.setSel(p, m.lists[p].sel+d)
}

func (m model) focusPane(p paneID) (model, tea.Cmd) {
	m.focus = p
	m.filterEditing = false
	if p == paneDiff || p == m.lastLeft {
		return m, nil
	}
	m.lastLeft = p
	var cmds []tea.Cmd
	if !m.animating && m.height > 0 {
		m.animating = true
		cmds = append(cmds, animTick())
	}
	var c tea.Cmd
	m, c = m.selectChanged(false, false)
	return m, tea.Batch(append(cmds, c)...)
}

// ---------------------------------------------------------------- data

func (m model) onStatus(msg statusMsg) (tea.Model, tea.Cmd) {
	if msg.err != nil {
		if msg.err.Error() != m.statusErr {
			m.statusErr = msg.err.Error()
			return m.showToast("status: "+m.statusErr, false)
		}
		return m, nil
	}
	m.statusErr = ""
	prev, had := m.selectedEntry()
	m.status = msg.st
	m.loaded = true
	if had {
		vis := m.visible(paneStatus)
		best := -1
		for i, idx := range vis {
			e := m.status.Entries[idx]
			if e.Path == prev.Path && e.Staged == prev.Staged {
				best = i
				break
			}
			if e.Path == prev.Path && best < 0 {
				best = i
			}
		}
		if best >= 0 {
			m.lists[paneStatus].sel = best
		}
	}
	m = m.clampList(paneStatus).storeOffset(paneStatus)
	if m.lastLeft == paneStatus {
		mm, c := m.selectChanged(true, true)
		return mm, c
	}
	return m, nil
}

func (m model) onBranches(msg branchesMsg) (tea.Model, tea.Cmd) {
	if msg.err != nil {
		return m, nil
	}
	prev, had := m.selectedBranch()
	m.branches = msg.branches
	if had {
		for i, idx := range m.visible(paneBranches) {
			if m.branches[idx].Name == prev.Name {
				m.lists[paneBranches].sel = i
				break
			}
		}
	}
	m = m.clampList(paneBranches).storeOffset(paneBranches)
	if m.lastLeft == paneBranches {
		mm, c := m.selectChanged(true, true)
		return mm, c
	}
	return m, nil
}

func (m model) onCommits(msg commitsMsg) (tea.Model, tea.Cmd) {
	if msg.err != nil {
		return m.showToast("log: "+msg.err.Error(), false)
	}
	prev, had := m.selectedCommit()
	m.commits = msg.commits
	if had {
		for i, idx := range m.visible(paneCommits) {
			if m.commits[idx].SHA == prev.SHA {
				m.lists[paneCommits].sel = i
				break
			}
		}
	}
	m = m.clampList(paneCommits).storeOffset(paneCommits)
	if m.lastLeft == paneCommits {
		mm, c := m.selectChanged(false, false)
		return mm, c
	}
	return m, nil
}

func (m model) onStashes(msg stashesMsg) (tea.Model, tea.Cmd) {
	if msg.err != nil {
		return m, nil
	}
	m.stashes = msg.stashes
	m = m.clampList(paneStash).storeOffset(paneStash)
	if m.lastLeft == paneStash {
		mm, c := m.selectChanged(false, false)
		return mm, c
	}
	return m, nil
}

func (m model) onDiff(msg diffMsg) (tea.Model, tea.Cmd) {
	if msg.seq != m.diffSeq {
		return m, nil
	}
	same := msg.key == m.diff.key
	m.diff = msg
	m.diffLoading = false
	m = m.rerenderDiff()
	if !same {
		m.vp.GotoTop()
	}
	return m, nil
}

func (m model) onAction(msg actionMsg) (tea.Model, tea.Cmd) {
	m.busy = maxInt(0, m.busy-1)
	m.lastRefresh = time.Now()
	var mm tea.Model
	var toastCmd tea.Cmd
	if msg.err != nil {
		mm, toastCmd = m.showToast(msg.verb+" failed: "+msg.err.Error(), false)
	} else {
		text := msg.ok
		if msg.verb == "commit" {
			text = "committed " + msg.out
		}
		mm, toastCmd = m.showToast(text, true)
	}
	m = mm.(model)
	return m, tea.Batch(toastCmd, m.refreshAll())
}

func (m model) showToast(text string, ok bool) (tea.Model, tea.Cmd) {
	m.toastID++
	id := m.toastID
	m.toast = &toast{id: id, text: text, ok: ok}
	return m, tea.Tick(toastLife, func(time.Time) tea.Msg { return toastExpireMsg{id: id} })
}

func (m model) runAction(verb, ok string, argSets ...[]string) (tea.Model, tea.Cmd) {
	if len(argSets) > 0 {
		m.lastGit = argSets[0]
	}
	m.busy++
	m, c := m.spinStart()
	return m, tea.Batch(gitAction(m.root, verb, ok, argSets...), c)
}

// ---------------------------------------------------------------- keys

func (m model) handleKey(k tea.KeyMsg) (tea.Model, tea.Cmd) {
	key := k.String()
	if m.noRepo {
		switch key {
		case "q", "ctrl+c", "esc", "enter":
			return m, tea.Quit
		}
		return m, nil
	}
	if m.modal != modalNone {
		if key == "esc" {
			return m.closeModal(), nil
		}
		return m.updateForm(k)
	}
	if m.showHelp {
		switch key {
		case "?", "esc", "q", "enter", "ctrl+c":
			m.showHelp = false
		}
		return m, nil
	}
	if m.filterEditing {
		return m.handleFilterKey(k)
	}
	switch key {
	case "ctrl+c", "q":
		return m, tea.Quit
	case "?":
		m.showHelp = true
		return m, nil
	case "r", "f5":
		m.lastRefresh = time.Now()
		mm, c := m.selectChanged(true, false)
		return mm, tea.Batch(mm.refreshAll(), c)
	case "tab":
		return m.focusPane((m.focus + 1) % paneCount)
	case "shift+tab":
		return m.focusPane((m.focus + paneCount - 1) % paneCount)
	case "1", "2", "3", "4":
		return m.focusPane(paneID(key[0] - '1'))
	case "h", "left":
		if m.focus == paneDiff {
			return m.focusPane(m.lastLeft)
		}
		return m, nil
	case "l", "right":
		if m.focus != paneDiff {
			return m.focusPane(paneDiff)
		}
		return m, nil
	case "/":
		mm, c := m.focusPane(m.lastLeft)
		mm.filterEditing = true
		return mm, c
	case "esc":
		if m.lists[m.lastLeft].filter != "" {
			m.lists[m.lastLeft].filter = ""
			m = m.clampList(m.lastLeft).storeOffset(m.lastLeft)
			return m.selectChanged(false, false)
		}
		return m, nil
	case "c":
		return m.openCommitForm()
	}
	if m.focus == paneDiff {
		return m.handleDiffKey(key)
	}
	return m.handleListKey(key)
}

func (m model) handleDiffKey(key string) (tea.Model, tea.Cmd) {
	switch key {
	case "j", "down":
		m.vp.LineDown(1)
	case "k", "up":
		m.vp.LineUp(1)
	case "ctrl+d", "pgdown":
		m.vp.HalfViewDown()
	case "ctrl+u", "pgup":
		m.vp.HalfViewUp()
	case "g", "home":
		m.vp.GotoTop()
	case "G", "end":
		m.vp.GotoBottom()
	}
	return m, nil
}

func (m model) handleListKey(key string) (tea.Model, tea.Cmd) {
	p := m.focus
	switch key {
	case "j", "down":
		return m.moveSel(p, 1)
	case "k", "up":
		return m.moveSel(p, -1)
	case "pgdown":
		return m.moveSel(p, 10)
	case "pgup":
		return m.moveSel(p, -10)
	case "g", "home":
		return m.setSel(p, 0)
	case "G", "end":
		return m.setSel(p, len(m.visible(p))-1)
	}
	switch p {
	case paneStatus:
		return m.handleStatusKey(key)
	case paneBranches:
		return m.handleBranchKey(key)
	case paneCommits:
		return m.handleCommitKey(key)
	case paneStash:
		return m.handleStashKey(key)
	}
	return m, nil
}

func (m model) handleStatusKey(key string) (tea.Model, tea.Cmd) {
	switch key {
	case " ":
		e, ok := m.selectedEntry()
		if !ok {
			return m, nil
		}
		args := stageToggleArgs(e)
		if e.Staged {
			if m.status.Unborn {
				args = []string{"reset", "-q", "--", e.Path}
			}
			return m.runAction("unstage", "unstaged "+e.Path, args)
		}
		return m.runAction("stage", "staged "+e.Path, args)
	case "a":
		if !m.hasEntries(false) {
			return m.showToast("nothing to stage", false)
		}
		return m.runAction("stage all", "staged all changes", []string{"add", "-A"})
	case "A":
		if !m.hasEntries(true) {
			return m.showToast("nothing staged", false)
		}
		return m.runAction("unstage all", "unstaged all changes", []string{"reset", "-q"})
	case "d":
		e, ok := m.selectedEntry()
		if !ok {
			return m, nil
		}
		m.confirm = confirmDiscard
		m.confirmEntry = e
		desc := "This permanently reverts the working-tree changes."
		if e.Code == "?" {
			desc = "This permanently deletes the untracked file."
		}
		return m.openConfirm("Discard changes to "+e.Path+"?", desc)
	case "s":
		if len(m.status.Entries) == 0 {
			return m.showToast("nothing to stash", false)
		}
		m.confirm = confirmStash
		return m.openConfirm("Stash all changes?", "Runs git stash push --include-untracked.")
	case "enter":
		if _, ok := m.selectedEntry(); ok {
			return m.focusPane(paneDiff)
		}
	}
	return m, nil
}

func (m model) hasEntries(staged bool) bool {
	for _, e := range m.status.Entries {
		if e.Staged == staged {
			return true
		}
	}
	return false
}

func (m model) handleBranchKey(key string) (tea.Model, tea.Cmd) {
	switch key {
	case "enter":
		b, ok := m.selectedBranch()
		if !ok {
			return m, nil
		}
		if b.Current {
			return m.showToast("already on "+b.Name, false)
		}
		return m.runAction("checkout", "switched to "+b.Name, []string{"checkout", b.Name})
	case "n":
		return m.openBranchForm()
	}
	return m, nil
}

func (m model) handleCommitKey(key string) (tea.Model, tea.Cmd) {
	switch key {
	case "enter":
		if _, ok := m.selectedCommit(); !ok {
			return m, nil
		}
		mm, c := m.selectChanged(true, false)
		mm, c2 := mm.focusPane(paneDiff)
		return mm, tea.Batch(c, c2)
	case "y":
		c, ok := m.selectedCommit()
		if !ok {
			return m, nil
		}
		return m, copyToClipboard(c.SHA)
	}
	return m, nil
}

func copyToClipboard(sha string) tea.Cmd {
	return func() tea.Msg {
		_, err := os.Stdout.WriteString(osc52.New(sha).Clipboard(osc52.SystemClipboard).String())
		return copiedMsg{sha: sha[:minInt(7, len(sha))], err: err}
	}
}

func (m model) handleStashKey(key string) (tea.Model, tea.Cmd) {
	if key == "enter" {
		s, ok := m.selectedStash()
		if !ok {
			return m, nil
		}
		return m.runAction("pop", "popped "+s.Ref, []string{"stash", "pop", s.Ref})
	}
	return m, nil
}

func (m model) handleFilterKey(k tea.KeyMsg) (tea.Model, tea.Cmd) {
	p := m.lastLeft
	switch k.String() {
	case "esc":
		m.lists[p].filter = ""
		m.filterEditing = false
	case "enter":
		m.filterEditing = false
		return m, nil
	case "backspace":
		f := []rune(m.lists[p].filter)
		if len(f) > 0 {
			m.lists[p].filter = string(f[:len(f)-1])
		}
	case "up":
		return m.moveSel(p, -1)
	case "down":
		return m.moveSel(p, 1)
	case "ctrl+c":
		return m, tea.Quit
	default:
		if k.Type == tea.KeyRunes || k.Type == tea.KeySpace {
			m.lists[p].filter += string(k.Runes)
		} else {
			return m, nil
		}
	}
	m.lists[p].sel = 0
	m = m.clampList(p).storeOffset(p)
	return m.selectChanged(false, false)
}

// ---------------------------------------------------------------- mouse

func (m model) handleMouse(e tea.MouseMsg) (tea.Model, tea.Cmd) {
	if m.noRepo || m.modal != modalNone || m.showHelp || e.Action != tea.MouseActionPress {
		return m, nil
	}
	l := m.layout()
	p, row, ok := l.hit(e.X, e.Y)
	if !ok {
		return m, nil
	}
	switch e.Button {
	case tea.MouseButtonWheelUp, tea.MouseButtonWheelDown:
		down := e.Button == tea.MouseButtonWheelDown
		if p == paneDiff {
			if down {
				m.vp.LineDown(3)
			} else {
				m.vp.LineUp(3)
			}
			return m, nil
		}
		if p != m.lastLeft {
			return m, nil
		}
		if down {
			return m.moveSel(p, 1)
		}
		return m.moveSel(p, -1)
	case tea.MouseButtonLeft:
		if p == paneDiff {
			return m.focusPane(paneDiff)
		}
		wasActive := p == m.lastLeft
		mm, c := m.focusPane(p)
		if !wasActive || row < 0 {
			return mm, c
		}
		rows, cursor := mm.rows(p, l.leftW-2)
		off := viewOffset(mm.lists[p].off, cursor, rows, l.paneH[p]-2)
		if r := off + row; r < len(rows) && rows[r].idx >= 0 {
			return mm.setSel(p, rows[r].idx)
		}
		return mm, c
	}
	return m, nil
}

// ---------------------------------------------------------------- modals

func (m model) modalWidth() int {
	return clampInt(m.width-10, 40, 66)
}

func (m model) openCommitForm() (tea.Model, tea.Cmd) {
	if !m.hasEntries(true) {
		return m.showToast("nothing staged", false)
	}
	fv := &formValues{}
	m.fv = fv
	m.form = huh.NewForm(huh.NewGroup(
		huh.NewInput().Key("title").Title("Commit message").
			Placeholder("feat: describe the change").Value(&fv.title).
			Validate(func(s string) error {
				if strings.TrimSpace(s) == "" {
					return errors.New("a message is required")
				}
				return nil
			}),
		huh.NewText().Key("body").Title("Description (optional)").
			Placeholder("ctrl+j inserts a newline").Lines(3).Value(&fv.body),
		huh.NewConfirm().Key("ok").Title("Create commit?").
			Affirmative("Commit").Negative("Cancel").Value(&fv.yes),
	)).WithTheme(huhTheme()).WithWidth(m.modalWidth()).WithShowHelp(true)
	m.modal = modalCommit
	m.modalTitle = "Commit staged changes"
	return m, m.form.Init()
}

func (m model) openBranchForm() (tea.Model, tea.Cmd) {
	fv := &formValues{}
	m.fv = fv
	m.form = huh.NewForm(huh.NewGroup(
		huh.NewInput().Key("branch").Title("New branch name").
			Placeholder("feature/my-change").Value(&fv.branch).
			Validate(func(s string) error {
				s = strings.TrimSpace(s)
				if s == "" {
					return errors.New("a name is required")
				}
				if strings.ContainsAny(s, " ~^:?*[\\") {
					return errors.New("invalid characters in branch name")
				}
				return nil
			}),
	)).WithTheme(huhTheme()).WithWidth(m.modalWidth()).WithShowHelp(true)
	m.modal = modalBranch
	m.modalTitle = "Create and checkout branch"
	return m, m.form.Init()
}

func (m model) openConfirm(question, desc string) (tea.Model, tea.Cmd) {
	fv := &formValues{}
	m.fv = fv
	m.form = huh.NewForm(huh.NewGroup(
		huh.NewConfirm().Key("yes").Title(question).Description(desc).
			Affirmative("Yes").Negative("No").Value(&fv.yes),
	)).WithTheme(huhTheme()).WithWidth(m.modalWidth()).WithShowHelp(true)
	m.modal = modalConfirm
	m.modalTitle = "Confirm"
	return m, m.form.Init()
}

func (m model) closeModal() model {
	m.modal = modalNone
	m.form = nil
	return m
}

func (m model) updateForm(msg tea.Msg) (tea.Model, tea.Cmd) {
	f, cmd := m.form.Update(msg)
	if ff, ok := f.(*huh.Form); ok {
		m.form = ff
	}
	switch m.form.State {
	case huh.StateAborted:
		return m.closeModal(), nil
	case huh.StateCompleted:
		return m.completeModal()
	}
	return m, cmd
}

func (m model) completeModal() (tea.Model, tea.Cmd) {
	kind := m.modal
	fv := m.fv
	m = m.closeModal()
	if fv == nil {
		return m, nil
	}
	switch kind {
	case modalCommit:
		if !fv.yes {
			return m, nil
		}
		args := []string{"commit", "-m", strings.TrimSpace(fv.title)}
		if body := strings.TrimSpace(fv.body); body != "" {
			args = append(args, "-m", body)
		}
		return m.runAction("commit", "", args, []string{"rev-parse", "--short", "HEAD"})
	case modalBranch:
		name := strings.TrimSpace(fv.branch)
		if name == "" {
			return m, nil
		}
		return m.runAction("branch", "created "+name, []string{"checkout", "-b", name})
	case modalConfirm:
		if !fv.yes {
			return m, nil
		}
		switch m.confirm {
		case confirmDiscard:
			e := m.confirmEntry
			if e.Code == "?" {
				m.busy++
				m.lastGit = []string{"rm", "-rf", "--", e.Path}
				mm, c := m.spinStart()
				return mm, tea.Batch(removePath(m.root, e.Path, "deleted "+e.Path), c)
			}
			argSets := discardArgSets(e)
			if e.Staged && m.status.Unborn {
				argSets = [][]string{{"reset", "-q", "--", e.Path}} // restore --staged needs a HEAD
			}
			return m.runAction("discard", "discarded "+e.Path, argSets...)
		case confirmStash:
			return m.runAction("stash", "stashed working tree", []string{"stash", "push", "--include-untracked"})
		}
	}
	return m, nil
}
